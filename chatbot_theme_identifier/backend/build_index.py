import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Your data folders
CLUSTERED_DOCS_PATH = "backend/data/clustered_docs"
INDEX_DIR = "backend/data/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

def build_faiss_index():
    all_texts = []
    metadata = []

    # Load all JSON chunks from clustered_docs
    for filename in os.listdir(CLUSTERED_DOCS_PATH):
        if filename.endswith(".json"):
            path = os.path.join(CLUSTERED_DOCS_PATH, filename)
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for chunk in chunks:
                text = chunk.get("text")
                if isinstance(text, dict):
                    text = text.get("text", "")
                all_texts.append(text)
                metadata.append(chunk)

    if not all_texts:
        print("[X] No texts found to index.")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "index_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("[OK] FAISS index and metadata saved.")

if __name__ == "__main__":
    build_faiss_index()
