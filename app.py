import os
import uuid
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from groq import Groq
import pypdf

app = Flask(__name__)

UPLOAD_FOLDER = "/data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)

# Simple in‑memory storage: filename -> full text
documents = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf'}

def extract_text(filepath, filename):
    if filename.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif filename.endswith('.pdf'):
        reader = pypdf.PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    return ""

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text(filepath, filename)
    if not text:
        return jsonify({"error": "Could not extract text"}), 500

    documents[filename] = text
    return jsonify({"message": f"Uploaded {filename}, length {len(text)} chars"}), 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    question = data['question']

    # If no documents, fallback to LLM only
    if not documents:
        prompt = question
    else:
        # Simple concatenation of all documents (for small knowledge bases)
        all_text = "\n\n".join(documents.values())
        prompt = f"""Answer the question based ONLY on the following text. If the answer is not in the text, say "I don't know".

Text:
{all_text[:4000]}   # limit to avoid token overflow

Question: {question}
Answer:"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = completion.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return jsonify({"answer": "I'm having trouble thinking right now."}), 500

@app.route('/')
def home():
    return jsonify({"status": "JARVIS with simplified Knowledge Base", "version": "layer20-simple"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
