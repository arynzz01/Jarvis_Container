import os
import uuid
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from groq import Groq
import pypdf

app = Flask(__name__)

# ------------------------------
# Configuration
# ------------------------------
UPLOAD_FOLDER = "/data/uploads"
SCHEDULE_FILE = "/data/schedules.json"
CHROMA_PATH = "/data/chroma"          # Not used in simplified version, but kept for future
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(SCHEDULE_FILE), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)

# In‑memory document storage (simplified RAG)
documents = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

# ------------------------------
# Helper functions (Layer 20)
# ------------------------------
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

# ------------------------------
# Proactive Scheduler (Layer 21)
# ------------------------------
def load_schedules():
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_schedules(schedules):
    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(schedules, f, indent=2)

def check_schedules():
    now = datetime.now().strftime("%H:%M")
    schedules = load_schedules()
    for task in schedules:
        if task.get("time") == now and not task.get("triggered_today", False):
            logger.info(f"🔔 PROACTIVE: {task.get('message', 'Scheduled task')}")
            task["triggered_today"] = True
            save_schedules(schedules)

def reset_daily_flags():
    schedules = load_schedules()
    for task in schedules:
        task["triggered_today"] = False
    save_schedules(schedules)

scheduler = BackgroundScheduler()
scheduler.add_job(func=check_schedules, trigger="interval", minutes=1)
scheduler.add_job(func=reset_daily_flags, trigger="cron", hour=0, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ------------------------------
# API Endpoints
# ------------------------------
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

    if not documents:
        prompt = question
    else:
        all_text = "\n\n".join(documents.values())
        prompt = f"""Answer based ONLY on the following text. If not in text, say "I don't know".

Text:
{all_text[:4000]}

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

@app.route('/add_task', methods=['POST'])
def add_task():
    data = request.get_json()
    if not data or 'time' not in data or 'message' not in data:
        return jsonify({"error": "Missing 'time' (HH:MM) or 'message'"}), 400
    schedules = load_schedules()
    schedules.append({
        "time": data['time'],
        "message": data['message'],
        "triggered_today": False
    })
    save_schedules(schedules)
    return jsonify({"message": f"Task added at {data['time']}: {data['message']}"}), 200

@app.route('/tasks', methods=['GET'])
def list_tasks():
    return jsonify(load_schedules())

@app.route('/')
def home():
    return jsonify({"status": "JARVIS with Knowledge Base & Proactive Suggestions", "version": "layer20+21"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
