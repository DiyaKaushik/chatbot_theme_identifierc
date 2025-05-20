from flask import Flask, render_template, request, jsonify
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from chatbot import get_chatbot_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query", "")
    if not query_text.strip():
        return jsonify({"error": "Empty query"}), 400
    
    result = get_chatbot_response(query_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
