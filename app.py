import os
import uuid
import json
import logging
import ast
import importlib.util
import sys
import traceback
import time
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
TOOLS_DIR = "/data/tools"
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(SCHEDULE_FILE), exist_ok=True)
os.makedirs(TOOLS_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)

# In-memory document storage (simplified RAG)
documents = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

# ------------------------------
# Wake word state (Layer 23)
# ------------------------------
awake = False
last_awake_time = 0
AWAKE_TIMEOUT = 10  # seconds

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
# Tool Creation (Layer 22)
# ------------------------------
def generate_tool_code(description, tool_name):
    prompt = f"""Write a Python function named '{tool_name}' that does: {description}
The function should take appropriate parameters and return a result. Do not include extra text, only the function definition. Use proper indentation."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        code = completion.choices[0].message.content
        if 'def' not in code:
            return None, "Generated code does not contain a function definition."
        return code, None
    except Exception as e:
        return None, str(e)

def save_tool(tool_name, code):
    filepath = os.path.join(TOOLS_DIR, f"{tool_name}.py")
    with open(filepath, 'w') as f:
        f.write(code)
    return filepath

def load_tool(tool_name):
    filepath = os.path.join(TOOLS_DIR, f"{tool_name}.py")
    if not os.path.exists(filepath):
        return None
    spec = importlib.util.spec_from_file_location(tool_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tool_name] = module
    spec.loader.exec_module(module)
    return getattr(module, tool_name, None)

# ------------------------------
# Mobile-friendly frontend (Layer 24)
# ------------------------------
FRONTEND_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>JARVIS – Mobile AI Assistant</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0a0e1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2rem;
            text-align: center;
            margin-bottom: 1rem;
            color: #4c9aff;
        }
        .card {
            background: #1e2433;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .card h2 {
            margin-top: 0;
            font-size: 1.3rem;
            color: #4c9aff;
        }
        textarea, input, button {
            width: 100%;
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 12px;
            border: none;
            font-size: 1rem;
        }
        textarea, input {
            background: #0f121c;
            color: #fff;
            border: 1px solid #2a3345;
        }
        button {
            background: #4c9aff;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:active {
            opacity: 0.7;
        }
        .response {
            background: #0f121c;
            padding: 12px;
            border-radius: 12px;
            margin-top: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            border-left: 4px solid #4c9aff;
        }
        .tool-item {
            background: #0f121c;
            margin: 8px 0;
            padding: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .tool-name {
            font-weight: bold;
        }
        .small-btn {
            width: auto;
            padding: 6px 12px;
            background: #2a3345;
        }
        hr {
            border-color: #2a3345;
        }
        .footer {
            text-align: center;
            font-size: 0.8rem;
            color: #6c7a8a;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🤖 JARVIS</h1>

    <!-- Chat / Ask -->
    <div class="card">
        <h2>💬 Ask JARVIS</h2>
        <textarea id="question" rows="3" placeholder="Type your question here..."></textarea>
        <button id="askBtn">Send</button>
        <div id="askResponse" class="response"></div>
    </div>

    <!-- Upload Document -->
    <div class="card">
        <h2>📄 Upload Document (TXT/PDF)</h2>
        <input type="file" id="fileInput" accept=".txt,.pdf">
        <button id="uploadBtn">Upload</button>
        <div id="uploadResponse" class="response"></div>
    </div>

    <!-- Tool Creation -->
    <div class="card">
        <h2>🛠️ Create a Tool</h2>
        <input type="text" id="toolName" placeholder="Tool name (e.g., celsius_to_fahrenheit)">
        <textarea id="toolDesc" rows="2" placeholder="Description (e.g., convert Celsius to Fahrenheit)"></textarea>
        <button id="createToolBtn">Create Tool</button>
        <div id="createToolResponse" class="response"></div>
    </div>

    <!-- List Tools -->
    <div class="card">
        <h2>📋 Available Tools</h2>
        <button id="listToolsBtn">Refresh Tools</button>
        <div id="toolsList" class="response"></div>
    </div>

    <!-- Use Tool -->
    <div class="card">
        <h2>🔧 Use a Tool</h2>
        <input type="text" id="useToolName" placeholder="Tool name">
        <input type="text" id="useToolArgs" placeholder="Arguments (comma separated, e.g., 25)">
        <button id="useToolBtn">Execute Tool</button>
        <div id="useToolResponse" class="response"></div>
    </div>

    <div class="footer">JARVIS v24 – Mobile ready</div>
</div>

<script>
    const apiBase = '';

    async function fetchJSON(url, options) {
        const response = await fetch(url, options);
        return response.json();
    }

    document.getElementById('askBtn').onclick = async () => {
        const question = document.getElementById('question').value;
        if (!question) return;
        const responseDiv = document.getElementById('askResponse');
        responseDiv.innerHTML = 'Thinking...';
        try {
            const data = await fetchJSON('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            responseDiv.innerHTML = data.answer || data.error || 'No response';
        } catch (err) {
            responseDiv.innerHTML = 'Error: ' + err.message;
        }
    };

    document.getElementById('uploadBtn').onclick = async () => {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        const responseDiv = document.getElementById('uploadResponse');
        responseDiv.innerHTML = 'Uploading...';
        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            responseDiv.innerHTML = data.message || data.error || 'Upload complete';
        } catch (err) {
            responseDiv.innerHTML = 'Error: ' + err.message;
        }
    };

    document.getElementById('createToolBtn').onclick = async () => {
        const name = document.getElementById('toolName').value;
        const desc = document.getElementById('toolDesc').value;
        if (!name || !desc) return;
        const responseDiv = document.getElementById('createToolResponse');
        responseDiv.innerHTML = 'Creating...';
        try {
            const data = await fetchJSON('/create_tool', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description: desc })
            });
            responseDiv.innerHTML = data.message || data.error || 'Tool created';
        } catch (err) {
            responseDiv.innerHTML = 'Error: ' + err.message;
        }
    };

    document.getElementById('listToolsBtn').onclick = async () => {
        const responseDiv = document.getElementById('toolsList');
        responseDiv.innerHTML = 'Loading...';
        try {
            const data = await fetchJSON('/list_tools', { method: 'GET' });
            const tools = data.tools || [];
            if (tools.length === 0) {
                responseDiv.innerHTML = 'No tools yet.';
                return;
            }
            let html = '';
            for (const tool of tools) {
                html += `<div class="tool-item"><span class="tool-name">${tool}</span><button class="small-btn" onclick="fillToolName('${tool}')">Use</button></div>`;
            }
            responseDiv.innerHTML = html;
        } catch (err) {
            responseDiv.innerHTML = 'Error: ' + err.message;
        }
    };

    window.fillToolName = (name) => {
        document.getElementById('useToolName').value = name;
    };

    document.getElementById('useToolBtn').onclick = async () => {
        const name = document.getElementById('useToolName').value;
        const argsStr = document.getElementById('useToolArgs').value;
        if (!name) return;
        let args = [];
        if (argsStr.trim()) {
            args = argsStr.split(',').map(s => {
                let v = s.trim();
                if (!isNaN(v)) return Number(v);
                if (v === 'true') return true;
                if (v === 'false') return false;
                return v;
            });
        }
        const responseDiv = document.getElementById('useToolResponse');
        responseDiv.innerHTML = 'Executing...';
        try {
            const data = await fetchJSON('/use_tool', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, args })
            });
            responseDiv.innerHTML = data.result !== undefined ? `Result: ${data.result}` : (data.error || 'No result');
        } catch (err) {
            responseDiv.innerHTML = 'Error: ' + err.message;
        }
    };
</script>
</body>
</html>
'''

# ------------------------------
# API Endpoints
# ------------------------------
@app.route('/wake', methods=['POST'])
def wake():
    global awake, last_awake_time
    awake = True
    last_awake_time = time.time()
    return jsonify({"status": "JARVIS is now awake", "timeout": AWAKE_TIMEOUT})

@app.before_request
def check_wake():
    global awake, last_awake_time
    if request.endpoint == 'ask':
        if not awake:
            return jsonify({"error": "JARVIS is asleep. Call /wake first."}), 403
        if time.time() - last_awake_time > AWAKE_TIMEOUT:
            awake = False
            return jsonify({"error": "JARVIS fell asleep. Wake again."}), 403

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

@app.route('/create_tool', methods=['POST'])
def create_tool():
    data = request.get_json()
    if not data or 'name' not in data or 'description' not in data:
        return jsonify({"error": "Missing 'name' or 'description'"}), 400
    tool_name = data['name']
    description = data['description']
    
    code, error = generate_tool_code(description, tool_name)
    if error:
        return jsonify({"error": f"Code generation failed: {error}"}), 500
    
    save_tool(tool_name, code)
    return jsonify({"message": f"Tool '{tool_name}' created", "code": code}), 200

@app.route('/list_tools', methods=['GET'])
def list_tools():
    tools = [f.replace('.py', '') for f in os.listdir(TOOLS_DIR) if f.endswith('.py')]
    return jsonify({"tools": tools})

@app.route('/use_tool', methods=['POST'])
def use_tool():
    data = request.get_json()
    if not data or 'name' not in data or 'args' not in data:
        return jsonify({"error": "Missing 'name' or 'args'"}), 400
    tool_name = data['name']
    args = data['args']
    
    tool_func = load_tool(tool_name)
    if not tool_func:
        return jsonify({"error": f"Tool '{tool_name}' not found"}), 404
    
    try:
        result = tool_func(*args)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Execution failed: {str(e)}", "traceback": traceback.format_exc()}), 500

@app.route('/')
def home():
    return FRONTEND_HTML

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
