import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX_PATH = "backend/data/faiss_index/index.faiss"
META_PATH = "backend/data/faiss_index/index_metadata.json"

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("[X] FAISS index or metadata not found.")
        return None, None, None

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, metadata, model

def retrieve_chunks(query, index, metadata, model, top_k=5):
    vec = model.encode([query]).astype(np.float32)
    D, I = index.search(vec, top_k)
    results = []
    for idx in I[0]:
        chunk = metadata[idx]
        results.append(chunk)
    return results

def generate_answer_basic(query, retrieved_chunks):
    context = "\n".join([
        chunk.get("text", "") if isinstance(chunk.get("text"), str) else str(chunk.get("text")) 
        for chunk in retrieved_chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    generator = pipeline('text-generation', model='distilgpt2')
    response = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
    answer = response[0]['generated_text']
    answer = answer[len(prompt):].strip()
    return answer

# New function to be called from Flask
def get_chatbot_response(query):
    index, metadata, model = load_index()
    if index is None:
        return "[X] FAISS index or metadata not found."

    retrieved = retrieve_chunks(query, index, metadata, model)
    answer = generate_answer_basic(query, retrieved)

    # Format output nicely (you can customize this)
    result = {
        "retrieved": retrieved,
        "answer": answer
    }
    return result
