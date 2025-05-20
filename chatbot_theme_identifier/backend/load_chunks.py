import os
import json

def load_chunks():
    # Construct path to 'backend/data/extracted_texts'
    folder = os.path.join(os.path.dirname(__file__), "data", "extracted_texts")
    folder = os.path.abspath(folder)
    print(f"Loading chunks from: {folder}")

    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return []

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    print(f"Files found: {files}")

    chunks = []
    for filename in files:
        file_path = os.path.join(folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Example JSON structure handling:
                # If your JSON is a dict with a "text" key containing the content
                if isinstance(data, dict) and "text" in data:
                    text = data["text"]
                    chunks.append({
                        "doc_id": filename,
                        "text": text,
                        "page": 1
                    })

                # If your JSON is a list of text snippets (pages or paragraphs)
                elif isinstance(data, list):
                    for i, entry in enumerate(data):
                        chunks.append({
                            "doc_id": filename,
                            "text": entry,
                            "page": i + 1
                        })
                else:
                    print(f"Warning: Unrecognized JSON format in {filename}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Total chunks loaded: {len(chunks)}")
    return chunks

# Test run when this file is executed directly
if __name__ == "__main__":
    load_chunks()
