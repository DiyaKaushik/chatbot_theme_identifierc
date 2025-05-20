import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

THEMES_PATH = "backend/data/themes"
CLUSTERED_OUTPUT_PATH = "backend/data/clustered_docs"
os.makedirs(CLUSTERED_OUTPUT_PATH, exist_ok=True)

def cluster_all_documents(num_clusters=5):
    all_chunks = []

    for filename in os.listdir(THEMES_PATH):
        if filename.endswith(".json"):
            filepath = os.path.join(THEMES_PATH, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                # Extract plain text from the nested 'text' field
                text_content = chunk.get("text")
                if isinstance(text_content, dict):
                    text = text_content.get("text", "")
                else:
                    text = str(text_content)

                chunk["plain_text"] = text
                all_chunks.append(chunk)

    if not all_chunks:
        print("[X] No chunks found.")
        return

    texts = [chunk["plain_text"] for chunk in all_chunks]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    for chunk, label in zip(all_chunks, labels):
        chunk["cluster"] = int(label)

    # Save clusters separately
    for i in range(num_clusters):
        cluster_chunks = [chunk for chunk in all_chunks if chunk["cluster"] == i]
        out_path = os.path.join(CLUSTERED_OUTPUT_PATH, f"cluster_{i}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cluster_chunks, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved cluster {i} with {len(cluster_chunks)} chunks.")

    # Optional visualization
    # Optional visualization
    reduced = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(embeddings)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10")
    plt.title("Clusters for all documents")
    plt.show()



if __name__ == "__main__":
    cluster_all_documents()
