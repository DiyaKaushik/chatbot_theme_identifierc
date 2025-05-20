# 🧠 Wasserstoff AI Chatbot Internship Project

This project was built as part of the recruitment process for the **AI Intern (Generative AI)** role at **Wasserstoff**. It demonstrates the ability to process documents using OCR, extract themes, perform semantic search, and build a citation-aware chatbot. 
.

---

## 📌 Features

- ✅ **OCR + Text Extraction** from scanned PDFs
- ✅ **Theme Extraction** using transformer-based models
- ✅ **Document Clustering** by theme
- ✅ **FAISS-based Semantic Search**
- ✅ **Citation-aware Chatbot** (answers with source reference)
- ✅ **Flask Web Interface** to interact with the chatbot

---

## 🛠️ Tech Stack

- **Python**
- **Flask** (Web UI)
- **Transformers** (Hugging Face models)
- **FAISS** (Vector search)
- **PyMuPDF, Tesseract, pdf2image** (for OCR)
- **scikit-learn** (Clustering)

---

## 🗂️ Folder Structure

wasserstoff-ai-chatbot/
│
├── data/ # OCRed text and document files
├── src/ # Core logic: OCR, embedding, search, chatbot
├── templates/ # Flask HTML templates
├── static/ # Static files (CSS, JS)
├── requirements.txt # Dependencies
├── app.py # Flask entry point
└── README.md # You're reading it!

---

## 🚀 How to Run Locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2.Run the Flask app
python app.py
3.Open your browser at http://127.0.0.1:5000
##📚 Citation Feature

The chatbot retrieves relevant passages using vector similarity and displays the source document name/page in the response. This is useful for research and academic referencing.
## 🤝 Acknowledgments

This take-home assignment was submitted as part of the recruitment process for **Wasserstoff's AI Intern (Generative AI)** position. 


---

## 📧 Contact

Feel free to reach out via Diyakaushik027@gmail.com or connect on [LinkedIn] https://www.linkedin.com/in/diya-kaushik-652806265
