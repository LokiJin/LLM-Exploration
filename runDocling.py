import os
import json
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# --------------------------
# 1) Load Previously Seen Files
# --------------------------
SEEN_FILE = "seen_files.json"
META_FILE = "metadata.pkl"
INDEX_FILE = "kb_index.faiss"

if os.path.exists(SEEN_FILE):
    with open(SEEN_FILE, "r") as f:
        seen_files = set(json.load(f))
else:
    seen_files = set()

print(f"Already processed {len(seen_files)} files")

# --------------------------
# 2) Find New Files
# --------------------------
def all_files_in_folder(folder):
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            out.append(os.path.join(root, fn))
    return out

all_files = all_files_in_folder("kb")
new_files = [f for f in all_files if f not in seen_files]

if not new_files:
    print("No new files found. Exiting.")
    exit(0)

print(f"Found {len(new_files)} new files to process")

# --------------------------
# 3) Parse New Files with Docling
# --------------------------
loader = DoclingLoader(file_path=new_files)
new_docs = loader.load()
print(f"Parsed {len(new_docs)} new docs")

# --------------------------
# 4) Chunk New Docs
# --------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

split_new_docs = []
for doc in new_docs:
    for chunk in splitter.split_text(doc.page_content):
        split_new_docs.append(
            Document(
            page_content=chunk,
            metadata={
            "source": doc.metadata.get("source","unknown"),
            "text": chunk  
        }
    )
)
print(f"Created {len(split_new_docs)} chunks from new docs")

# --------------------------
# 5) Load or Init FAISS + Metadata
# --------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadatas = pickle.load(f)
    print("Loaded existing index and metadata")
else:
    index = None
    metadatas = []
    print("No existing index — starting fresh")

# --------------------------
# 6) Embed & Normalize New Chunks
# --------------------------
texts = [d.page_content for d in split_new_docs]
if texts:
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    # Normalize so inner product == cosine similarity
    faiss.normalize_L2(vectors)

    # ------------------------
    # 7) Update FAISS Index (Inner Product)
    # ------------------------
    dim = vectors.shape[1]

    if index is None:
        index = faiss.IndexFlatIP(dim)  # inner product index for cosine

    index.add(vectors)
    print(f"Index now has {index.ntotal} vectors")

    # append metadata
    metadatas.extend([d.metadata for d in split_new_docs])

    # ------------------------
    # 8) Save Updated Index + Metadata
    # ------------------------
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadatas, f)

    print("Saved updated index + metadata")

# --------------------------
# 9) Update Seen Files
# --------------------------
seen_files.update(new_files)
with open(SEEN_FILE, "w") as f:
    json.dump(list(seen_files), f)

print("Done! Updated seen file list.")
