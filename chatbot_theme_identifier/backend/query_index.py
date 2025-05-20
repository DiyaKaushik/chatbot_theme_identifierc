import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "backend/data/faiss_index/index.faiss"
META_PATH = "backend/data/faiss_index/index_metadata.json"

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("[X] FAISS index or metadata file not found.")
        return None, None, None

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata, SentenceTransformer('all-MiniLM-L6-v2')

def query_index(user_query, top_k=5):
    index, metadata, model = load_index()
    if index is None:
        return

    query_vec = model.encode([user_query]).astype(np.float32)
    D, I = index.search(query_vec, top_k)

    print(f"\n🔍 Top {top_k} Results for Query: \"{user_query}\"")
    for rank, idx in enumerate(I[0]):
        result = metadata[idx]
        doc_id = result.get("doc_id", "Unknown")
        page = result.get("page", "N/A")
        text = result.get("text", "")[:300].replace("\n", " ")  # Truncate
        print(f"\n{rank+1}. [Doc: {doc_id}, Page: {page}]\n{text}...\n")

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    query_index(user_input)
