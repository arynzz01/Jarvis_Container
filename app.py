import os
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import pypdf

app = Flask(__name__)

UPLOAD_FOLDER = "/data/uploads"
CHROMA_PATH = "/data/chroma"
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="knowledge_base")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

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

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}_{uuid.uuid4().hex}"
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedder.encode(chunk).tolist()],
            metadatas=[{"source": filename, "chunk_index": i}],
            documents=[chunk]
        )
    return jsonify({"message": f"Uploaded {filename}, created {len(chunks)} chunks"}), 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    question = data['question']

    question_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=3)
    contexts = results['documents'][0] if results['documents'] else []

    if contexts:
        context = "\n\n".join(contexts)
        prompt = f"""Answer based ONLY on the context. If not in context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    else:
        prompt = f"Answer: {question}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = completion.choices[0].message.content
        return jsonify({"answer": answer, "used_context": len(contexts) > 0})
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return jsonify({"answer": "I'm having trouble thinking right now."}), 500

@app.route('/')
def home():
    return jsonify({"status": "JARVIS with Knowledge Base", "version": "layer20"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
