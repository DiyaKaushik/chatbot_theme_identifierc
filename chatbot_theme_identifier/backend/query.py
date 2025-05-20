from load_chunks import load_chunks
from vector_store import VectorStore

print("Script started")

chunks = load_chunks()
print(f"Chunks loaded: {len(chunks)}")

store = VectorStore()
print("Building index...")
store.build_index(chunks)
print("Index built.")

print("Ready to take queries.")

while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        print("Exiting.")
        break

    results = store.search(query)
    if not results:
        print("No matches found.")
    else:
        print("\nTop Matches:\n")
        for res in results:
            print(f"[{res['doc_id']}] Page {res['page']}\n→ {res['text'][:300]}...\n")
