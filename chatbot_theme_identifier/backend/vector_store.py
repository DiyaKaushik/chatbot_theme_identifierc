from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, chunks):
    # Filter chunks to keep only those with string 'text' fields
     valid_chunks = [chunk for chunk in chunks if isinstance(chunk.get("text", None), str)]

     if not valid_chunks:
        print("No valid text chunks found to build the index.")
        return

    # Extract text strings from filtered chunks
     texts = [chunk["text"] for chunk in valid_chunks]

    # (Optional) Debug print to check types
     for i, t in enumerate(texts):
        print(f"Chunk {i} text type: {type(t)}; preview: {t[:100]}")

    # Encode texts to embeddings
     embeddings = self.model.encode(texts, show_progress_bar=True)

     embeddings = np.array(embeddings).astype('float32')

     self.index = faiss.IndexFlatL2(embeddings.shape[1])
     self.index.add(embeddings)

    # Save metadata matching filtered chunks
     self.metadata = valid_chunks

    def search(self, query, top_k=5):
        # Encode the query text
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Perform similarity search in the index
        D, I = self.index.search(query_embedding, top_k)

        # Retrieve the corresponding chunks using the indices
        results = [self.metadata[i] for i in I[0]]

        return results
