import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

conversation_memory = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis")

def sanitise_input(text):
    dangerous = [';', '--', '/*', '*/', 'exec', 'eval', '__import__']
    for d in dangerous:
        text = text.replace(d, '')
    if len(text) > 1000:
        text = text[:1000]
    return ''.join(c for c in text if c.isprintable())

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def llm_chat(user_message, conversation_history):
    messages = [{"role": "system", "content": "You are JARVIS, a helpful AI assistant."}]
    messages.extend(conversation_history[-10:])
    messages.append({"role": "user", "content": user_message})
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",   # ✅ known working model
        "messages": messages,
        "temperature": 0.7
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            logger.error(f"Groq API error {response.status_code}: {response.text}")
            return f"I'm having trouble thinking right now. (API error {response.status_code})"
        data = response.json()
        reply = data['choices'][0]['message']['content']
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "I'm having trouble thinking right now."

def web_search(query):
    if not TAVILY_API_KEY:
        return "Web search not configured (no API key)."
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 2},
            timeout=30
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No results found."
        summaries = [f"• {r['title']}: {r['content'][:150]}..." for r in results]
        return "\n".join(summaries)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Web search failed."

@app.route('/')
def home():
    return jsonify({"status": "JARVIS is running", "version": "integrated-0.2"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    raw_question = data['question']
    question = sanitise_input(raw_question)
    logger.info(f"Question: {question}")

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
