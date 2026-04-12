
import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from groq import Groq
import requests

# ------------------------------
# 1. Configuration & Security
# ------------------------------
app = Flask(__name__)

# Load API keys from environment variables (set them in your deployment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# Simple in‑memory conversation memory (for demo)
conversation_memory = []

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

# Input sanitisation
def sanitise_input(text):
    dangerous = [';', '--', '/*', '*/', 'exec', 'eval', '__import__']
    for d in dangerous:
        text = text.replace(d, '')
    if len(text) > 1000:
        text = text[:1000]
    return ''.join(c for c in text if c.isprintable())

# ------------------------------
# 2. LLM Core (Groq)
# ------------------------------
client = Groq(api_key=GROQ_API_KEY)

def llm_chat(user_message, conversation_history):
    messages = [{"role": "system", "content": "You are JARVIS, a helpful AI assistant."}]
    messages.extend(conversation_history[-10:])  # keep last 10 exchanges
    messages.append({"role": "user", "content": user_message})
    try:
        completion = client.chat.completions.create(
            model="llama3-3-70b-versatile",
            messages=messages,
            temperature=0.7
        )
        reply = completion.choices[0].message.content
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "I'm having trouble thinking right now."

# ------------------------------
# 3. Web Search (Tavily)
# ------------------------------
def web_search(query):
    if not TAVILY_API_KEY:
        return "Web search not configured (no API key)."
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 2}
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No results found."
        summaries = [f"• {r['title']}: {r['content'][:150]}..." for r in results]
        return "
".join(summaries)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Web search failed."

# ------------------------------
# 4. Flask Routes
# ------------------------------
@app.route('/')
def home():
    return jsonify({"status": "JARVIS is running", "version": "integrated-0.1"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    raw_question = data['question']
    question = sanitise_input(raw_question)
    logger.info(f"Question: {question}")

    # Optional: detect tool use
    if "search" in question.lower():
        result = web_search(question)
        return jsonify({"answer": result})
    else:
        answer = llm_chat(question, conversation_memory)
        return jsonify({"answer": answer})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_length": len(conversation_memory)
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "groq_api_configured": bool(GROQ_API_KEY),
        "tavily_api_configured": bool(TAVILY_API_KEY),
        "logging": "active",
        "security": "input sanitisation enabled"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
