import os
import json
import logging
import time
import uuid
import importlib.util
import sys
import traceback
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
# Layer 25: JARVIS Asks Questions
# ------------------------------
QUESTIONS_FILE = "/data/pending_questions.json"

def load_pending_questions():
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_pending_questions(questions):
    with open(QUESTIONS_FILE, 'w') as f:
        json.dump(questions, f, indent=2)

def generate_proactive_question():
    context = ""
    if documents:
        sample_text = list(documents.values())[0][:500]
        context = f"Based on the following text: {sample_text}\n"
    prompt = f"""{context}Generate a single, interesting, open-ended question that a helpful AI assistant might ask the user to start a conversation. The question should be friendly and relevant. Do not include any extra text, just the question."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        question = completion.choices[0].message.content.strip()
        return question
    except Exception as e:
        logger.error(f"Failed to generate question: {e}")
        return "What would you like to talk about today?"

def scheduled_question_generation():
    if len(load_pending_questions()) < 5:
        question = generate_proactive_question()
        pending = load_pending_questions()
        pending.append({
            "timestamp": datetime.now().isoformat(),
            "question": question
        })
        save_pending_questions(pending)
        logger.info(f"Auto-generated question: {question}")

scheduler.add_job(func=scheduled_question_generation, trigger="interval", hours=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ------------------------------
# Layer 26: Infinite Memory
# ------------------------------
MEMORY_FILE = "/data/infinite_memory.json"

def load_infinite_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {"recent": [], "summaries": []}

def save_infinite_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def summarize_conversation(exchanges):
    if not exchanges:
        return ""
    text = "\n".join([f"{e['role']}: {e['content']}" for e in exchanges])
    prompt = f"""Summarize the following conversation into 2-3 short facts. Keep important information like names, preferences, and decisions.

Conversation:
{text}

Summary:"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return ""

def add_to_infinite_memory(role, content):
    memory = load_infinite_memory()
    memory["recent"].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    if len(memory["recent"]) > 6:
        to_summarize = memory["recent"][:3]
        summary = summarize_conversation(to_summarize)
        if summary:
            memory["summaries"].append({"timestamp": datetime.now().isoformat(), "summary": summary})
        memory["recent"] = memory["recent"][3:]
    save_infinite_memory(memory)

def get_infinite_memory_context():
    memory = load_infinite_memory()
    context = ""
    if memory["summaries"]:
        recent_summaries = memory["summaries"][-3:]
        context += "Long-term memories:\n" + "\n".join([s["summary"] for s in recent_summaries]) + "\n\n"
    if memory["recent"]:
        context += "Recent conversation:\n" + "\n".join([f"{e['role']}: {e['content']}" for e in memory["recent"]])
    return context

# ------------------------------
# Layer 27: Self-Evolving (Feedback & Prompt Improvement)
# ------------------------------
SYSTEM_PROMPT_FILE = "/data/system_prompt.txt"
FEEDBACK_FILE = "/data/feedback.json"

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are JARVIS, a helpful AI assistant. Be concise, accurate, and friendly."

def load_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, 'r') as f:
            return f.read().strip()
    else:
        with open(SYSTEM_PROMPT_FILE, 'w') as f:
            f.write(DEFAULT_SYSTEM_PROMPT)
        return DEFAULT_SYSTEM_PROMPT

def save_system_prompt(prompt):
    with open(SYSTEM_PROMPT_FILE, 'w') as f:
        f.write(prompt)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback(feedback_list):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_list, f, indent=2)

def improve_system_prompt():
    """Use LLM to generate a better system prompt based on past feedback and conversation samples."""
    feedback_list = load_feedback()
    if len(feedback_list) < 3:
        return False  # not enough feedback yet

    # Collect last 5 positive feedbacks (or all if less)
    positive = [f for f in feedback_list if f.get('rating') == 'good'][-5:]
    if not positive:
        return False

    # Sample conversations from memory? Could use recent exchanges from infinite memory.
    memory = load_infinite_memory()
    recent_exchanges = memory["recent"][-5:]  # last 5 exchanges
    conversation_text = "\n".join([f"{e['role']}: {e['content']}" for e in recent_exchanges])

    prompt = f"""You are an AI system that helps improve the system prompt for an AI assistant named JARVIS.
Current system prompt:
{load_system_prompt()}

Based on the following user feedback (ratings and comments) and recent conversation, suggest an improved system prompt.
The new prompt should be concise (2-3 sentences), guiding the assistant to be more helpful, accurate, and engaging.
Write only the new system prompt, nothing else.

Feedback:
{json.dumps(positive, indent=2)}

Recent conversation:
{conversation_text}

New system prompt:"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        new_prompt = completion.choices[0].message.content.strip()
        if new_prompt:
            save_system_prompt(new_prompt)
            logger.info(f"System prompt evolved to: {new_prompt}")
            return True
        return False
    except Exception as e:
        logger.error(f"Prompt improvement failed: {e}")
        return False

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data or 'rating' not in data:
        return jsonify({"error": "Missing 'rating' (good/bad)"}), 400
    rating = data['rating']
    comment = data.get('comment', '')
    # Store feedback
    fb = {
        "timestamp": datetime.now().isoformat(),
        "rating": rating,
        "comment": comment
    }
    feedback_list = load_feedback()
    feedback_list.append(fb)
    save_feedback(feedback_list)
    # Every 5 feedbacks, attempt to improve prompt
    if len([f for f in feedback_list if f.get('rating') == 'good']) % 5 == 0:
        improve_system_prompt()
    return jsonify({"message": "Feedback recorded"}), 200

# ------------------------------
# Layer 28: Predictive AI (Learn routines)
# ------------------------------
PREDICTIVE_DATA_FILE = "/data/predictive_data.json"

def load_predictive_data():
    if os.path.exists(PREDICTIVE_DATA_FILE):
        with open(PREDICTIVE_DATA_FILE, 'r') as f:
            return json.load(f)
    return {"logs": []}  # list of {"hour": int, "question": str, "timestamp": iso}

def save_predictive_data(data):
    with open(PREDICTIVE_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def log_question(question):
    """Log user question with hour of day for pattern learning."""
    data = load_predictive_data()
    data["logs"].append({
        "hour": datetime.now().hour,
        "question": question,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last 1000 logs to avoid unbounded growth
    if len(data["logs"]) > 1000:
        data["logs"] = data["logs"][-1000:]
    save_predictive_data(data)

def get_prediction():
    """Return the most common question asked at the current hour."""
    data = load_predictive_data()
    if not data["logs"]:
        return None
    current_hour = datetime.now().hour
    # Filter logs for this hour
    hour_logs = [entry["question"] for entry in data["logs"] if entry["hour"] == current_hour]
    if not hour_logs:
        return None
    # Find most frequent question
    from collections import Counter
    counter = Counter(hour_logs)
    most_common = counter.most_common(1)[0][0]
    return most_common

# ------------------------------
# Layer 29: Multi-Agent Debate
# ------------------------------
import random

DEBATE_ROUNDS = 2  # number of debate exchanges
AGENTS = [
    {"name": "Scientist", "prompt": "You are a rigorous, evidence‑driven scientist. Focus on facts, data, and logical consistency."},
    {"name": "Creative", "prompt": "You are a creative thinker. Explore unconventional ideas, analogies, and outside‑the‑box solutions."},
    {"name": "Skeptic", "prompt": "You are a skeptic. Question assumptions, identify flaws, and demand proof."}
]

def debate_round(question, previous_arguments):
    """Run one round of debate: each agent responds to the question and previous arguments."""
    responses = []
    for agent in AGENTS:
        # Build prompt including previous arguments
        context = f"Question: {question}\n"
        if previous_arguments:
            context += "Previous arguments:\n" + "\n".join([f"{a['agent']}: {a['response']}" for a in previous_arguments])
        else:
            context += "This is the first round. Provide your initial analysis.\n"
        full_prompt = f"{agent['prompt']}\n\n{context}\n\nYour response:"
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.5
            )
            response = completion.choices[0].message.content
            responses.append({"agent": agent['name'], "response": response})
        except Exception as e:
            logger.error(f"Debate agent {agent['name']} failed: {e}")
            responses.append({"agent": agent['name'], "response": "I'm unable to respond."})
    return responses

@app.route('/debate', methods=['POST'])
def debate():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    question = data['question']
    rounds = data.get('rounds', DEBATE_ROUNDS)

    all_responses = []
    for r in range(rounds):
        round_responses = debate_round(question, all_responses)
        all_responses.extend(round_responses)

    # Produce a final answer by asking a "judge" agent to synthesize the debate
    debate_transcript = "\n".join([f"{resp['agent']}: {resp['response']}" for resp in all_responses])
    synthesis_prompt = f"""You are a neutral judge. Based on the following debate, produce a concise, balanced, and accurate final answer to the question: "{question}".

Debate transcript:
{debate_transcript}

Final answer:"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3
        )
        final_answer = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        final_answer = "I'm having trouble reaching a consensus."

    return jsonify({
        "question": question,
        "rounds": rounds,
        "transcript": all_responses,
        "answer": final_answer
    })

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
    
    # Log for predictive AI
    log_question(question)
    
    # Add user question to infinite memory
    add_to_infinite_memory("user", question)
    
    # Build prompt: combine infinite memory context + documents + question
    memory_context = get_infinite_memory_context()
    doc_context = ""
    if documents:
        all_text = "\n\n".join(documents.values())
        doc_context = f"Document text:\n{all_text[:4000]}\n\n"
    
    if memory_context or doc_context:
        prompt = f"""{memory_context}{doc_context}User question: {question}\nAnswer based on the above context and your knowledge."""
    else:
        prompt = question
    
    system_prompt = load_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3
        )
        answer = completion.choices[0].message.content
        # Add assistant response to infinite memory
        add_to_infinite_memory("assistant", answer)
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

@app.route('/jarvis_ask', methods=['POST'])
def jarvis_ask():
    question = generate_proactive_question()
    pending = load_pending_questions()
    pending.append({
        "timestamp": datetime.now().isoformat(),
        "question": question
    })
    save_pending_questions(pending)
    return jsonify({"message": "Question generated", "question": question}), 200

@app.route('/pending_questions', methods=['GET'])
def get_pending_questions():
    return jsonify(load_pending_questions())

@app.route('/clear_questions', methods=['POST'])
def clear_questions():
    save_pending_questions([])
    return jsonify({"message": "All pending questions cleared"}), 200

@app.route('/predict', methods=['GET'])
def predict():
    prediction = get_prediction()
    if prediction:
        return jsonify({"prediction": prediction})
    else:
        return jsonify({"prediction": "Not enough data yet"})

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
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0a0e1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { font-size: 2rem; text-align: center; margin-bottom: 1rem; color: #4c9aff; }
        .card {
            background: #1e2433;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .card h2 { margin-top: 0; font-size: 1.3rem; color: #4c9aff; }
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
        button:active { opacity: 0.7; }
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
        .tool-name { font-weight: bold; }
        .small-btn { width: auto; padding: 6px 12px; background: #2a3345; }
        hr { border-color: #2a3345; }
        .footer { text-align: center; font-size: 0.8rem; color: #6c7a8a; }
    </style>
</head>
<body>
<div class="container">
    <h1>🤖 JARVIS</h1>
    <div class="card">
        <h2>💬 Ask JARVIS</h2>
        <textarea id="question" rows="3" placeholder="Type your question here..."></textarea>
        <button id="askBtn">Send</button>
        <div id="askResponse" class="response"></div>
    </div>
    <div class="card">
        <h2>📄 Upload Document (TXT/PDF)</h2>
        <input type="file" id="fileInput" accept=".txt,.pdf">
        <button id="uploadBtn">Upload</button>
        <div id="uploadResponse" class="response"></div>
    </div>
    <div class="card">
        <h2>🛠️ Create a Tool</h2>
        <input type="text" id="toolName" placeholder="Tool name (e.g., celsius_to_fahrenheit)">
        <textarea id="toolDesc" rows="2" placeholder="Description (e.g., convert Celsius to Fahrenheit)"></textarea>
        <button id="createToolBtn">Create Tool</button>
        <div id="createToolResponse" class="response"></div>
    </div>
    <div class="card">
        <h2>📋 Available Tools</h2>
        <button id="listToolsBtn">Refresh Tools</button>
        <div id="toolsList" class="response"></div>
    </div>
    <div class="card">
        <h2>🔧 Use a Tool</h2>
        <input type="text" id="useToolName" placeholder="Tool name">
        <input type="text" id="useToolArgs" placeholder="Arguments (comma separated, e.g., 25)">
        <button id="useToolBtn">Execute Tool</button>
        <div id="useToolResponse" class="response"></div>
    </div>
    <div class="footer">JARVIS v26 – Infinite Memory</div>
</div>
<script>
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

@app.route('/')
def home():
    return FRONTEND_HTML
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
