import os
import json
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

# ------------------------------
# Infinite Memory (Layer 26)
# ------------------------------
MEMORY_FILE = "/data/infinite_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {"recent": [], "summaries": []}

def save_memory(memory):
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

def add_to_memory(role, content):
    memory = load_memory()
    memory["recent"].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    # Trigger summarisation when recent exceeds 6 exchanges
    if len(memory["recent"]) > 6:
        to_summarize = memory["recent"][:3]          # summarise oldest 3
        summary = summarize_conversation(to_summarize)
        if summary:
            memory["summaries"].append({"timestamp": datetime.now().isoformat(), "summary": summary})
        # keep only the most recent 3 exchanges
        memory["recent"] = memory["recent"][3:]
    save_memory(memory)

def get_memory_context():
    memory = load_memory()
    context = ""
    if memory["summaries"]:
        # use the last 3 summaries
        recent_summaries = memory["summaries"][-3:]
        context += "Long-term memories:\n" + "\n".join([s["summary"] for s in recent_summaries]) + "\n\n"
    if memory["recent"]:
        context += "Recent conversation:\n" + "\n".join([f"{e['role']}: {e['content']}" for e in memory["recent"]])
    return context

# ------------------------------
# API Endpoints
# ------------------------------
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    question = data['question']

    # store user question
    add_to_memory("user", question)

    memory_context = get_memory_context()
    if memory_context:
        prompt = f"""{memory_context}\n\nUser question: {question}\nAnswer based on the above context and your knowledge."""
    else:
        prompt = question

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = completion.choices[0].message.content
        # store assistant response
        add_to_memory("assistant", answer)
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return jsonify({"answer": "I'm having trouble thinking right now."}), 500

@app.route('/')
def home():
    return jsonify({"status": "JARVIS with Infinite Memory", "version": "layer26"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
