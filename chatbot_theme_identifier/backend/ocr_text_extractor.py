import os
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import json

BASE_DIR = os.path.dirname(__file__)

RAW_DOCS_PATH = os.path.join(BASE_DIR, "data", "raw_docs")
EXTRACTED_TEXT_PATH = os.path.join(BASE_DIR, "data", "extracted_texts")

os.makedirs(RAW_DOCS_PATH, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_PATH, exist_ok=True)

def extract_text_from_pdf(file_path, doc_id):
    all_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    all_text.append({
                        "page": i + 1,
                        "text": text.strip(),
                        "type": "digital"
                    })
                else:
                    images = convert_from_path(file_path, first_page=i+1, last_page=i+1)
                    ocr_text = pytesseract.image_to_string(images[0])
                    all_text.append({
                        "page": i + 1,
                        "text": ocr_text.strip(),
                        "type": "ocr"
                    })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    output_path = os.path.join(EXTRACTED_TEXT_PATH, f"{doc_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_text, f, indent=2, ensure_ascii=False)

    print(f"[OK] Extracted: {file_path} -> {output_path}")


if __name__ == "__main__":
    for filename in os.listdir(RAW_DOCS_PATH):
        if filename.lower().endswith(".pdf"):
            doc_path = os.path.join(RAW_DOCS_PATH, filename)
            doc_id = os.path.splitext(filename)[0]
            extract_text_from_pdf(doc_path, doc_id)
