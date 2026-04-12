
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "JARVIS is running!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('question', '')
    response = f"JARVIS heard: {user_input}"
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
