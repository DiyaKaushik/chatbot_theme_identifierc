import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from load_chunks import load_chunks

# Get the absolute directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "themes")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_themes(n_clusters=5):
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    if not chunks:
        print("❌ No chunks found.")
        return

    texts = [chunk["text"] if isinstance(chunk["text"], str) else str(chunk["text"]) for chunk in chunks]


    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8)
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    clustered = {}
    for idx, label in enumerate(kmeans.labels_):
        clustered.setdefault(label, []).append(chunks[idx])

    for label, group in clustered.items():
        with open(os.path.join(OUTPUT_FOLDER, f"theme_{label}.json"), "w", encoding="utf-8") as f:
            json.dump(group, f, indent=2, ensure_ascii=False)

    print(f"[OK] Extracted {len(clustered)} themes into {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    extract_themes(n_clusters=5)
