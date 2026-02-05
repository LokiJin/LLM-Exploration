import os
import pickle
import requests
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# --------------------------
# 1) Load FAISS Index + Metadata
# --------------------------

INDEX_FILE = "kb_index.faiss"
META_FILE = "metadata.pkl"

cos_min_score = 0.3  # minimum cosine similarity for retrieved chunks

# Load FAISS index
index = faiss.read_index(INDEX_FILE)

# Load stored metadata
with open(META_FILE, "rb") as f:
    metadatas = pickle.load(f)

print(f"Loaded index with {index.ntotal} vectors")

# --------------------------
# 2) Load Embedding Model
# --------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# 3) Function to Retrieve Top K Chunks
# --------------------------

def retrieve(query, k, min_score):
    # Embed the query
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    
    # Normalize for cosine similarity indexing (inner product)
    faiss.normalize_L2(q_vec)
    
    # Search index
    D, I = index.search(q_vec, k)
    
    # Collect retrieved texts + scores
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(metadatas) and score >= min_score:
            source = metadatas[idx].get("source", "<unknown>")
            results.append({
                "score": float(score),
                "source": metadatas[idx]["source"],
                "text": metadatas[idx]["text"]
})
    return results

# --------------------------
# 4) Query Loop
# --------------------------

while True:
    prompt = input("\nEnter your question (or 'quit'): ").strip()
    if prompt.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    
    # Retrieve top context
    top_k = 5
    retrieved = retrieve(prompt, k=top_k, min_score=cos_min_score)
    
    if len(retrieved) == 0:
        print("No relevant documents found.")
        continue
    
    for item in retrieved:
        print(f"score={item['score']:.4f}  source={item['source']} chunk='{item['text']}...'")

    # Build context text
    context_text = ""
    for i, item in enumerate(retrieved, 1):
        context_text += f"[{i}]{item['text']}\n\n"
    
    # Create RAG‑style prompt
    rag_prompt = (
        "Use the following context to help answer the question, be cautious of everything you read.\n\n"
        "CONTEXT:\n" + context_text + "\n\n"
        "QUESTION:\n" + prompt + "\n\n"
        "Answer:\n"
    )
    
    # --------------------------
    # 5) Send to Local LLM Server
    # --------------------------
    
    llm_url = "http://127.0.0.1:8080/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
   
    payload = {
        "model": "gemma-3-12b-it-q4_0",  
        "messages": [
            {"role": "system", "content": "Use the following context to help answer the question, be cautious of everything you read."},
            {"role": "user", "content": rag_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(llm_url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text content from the assistant
        # Different servers might nest it differently, but most OpenAI‑compatible APIs do this:
        answer = data["choices"][0]["message"]["content"]
        
        print("\n--- Answer ---")
        print(answer)
        print("--------------")
    
    except requests.RequestException as e:
        print(f"Error calling LLM server: {e}")
        if response is not None:
            print(response.text)