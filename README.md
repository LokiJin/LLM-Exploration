Local RAG Knowledge Base

This project is meant to explore current and evolving features and use cases of LLMs including RAG, Agents, and more. The code is not meant for prod, but is useful for learning, configuring, and applying AI concepts.

Currently, it is setup to implement a Retrieval-Augmented Generation (RAG) pipeline.

Indexing: Scans kb/ folder, parses .docx files with Docling, splits into chunks, embeds with all-MiniLM-L6-v2, normalizes, and stores vectors in a FAISS index. Metadata (source + text) and seen files are saved via pickle/json.

Querying: Embeds user queries, retrieves top-K similar chunks via cosine similarity, builds a RAG-style prompt, and sends it to a local LLM server.

Features: Incremental updates, fully local, simple metadata mapping, extensible baseline for RAG applications.




FAQ
I don't want to use a local LLM
	-> Then you will need to adjust the LLM API url, request payload, and setup authentication. See your cloud LLM providers documentation.

I want to try another local LLM besides gemma-3-12b-it-q4_0.
	-> You can set up any compatible local model using llama.cpp, Ollama, or another local LLM server. Then just update payload["model"] to the new model’s name. Other models may have different required payload structures.

I am not on Windows 11
	-> You may need to re-configure the file directory look up appropriately for your OS

I want to use a different embedding model
	-> Swap out SentenceTransformer in both scripts.

I want to tune RAG settings
	-> For retrieval and embedding, check out 'splitter', 'model',
	-> For generation check out 'TOP_K', 'cos_min_score', 'rag_prompt', and 'payload' in ragtheLLM.py
