import os
import ast
import importlib.util
import sys
import traceback

TOOLS_DIR = "/data/tools"
os.makedirs(TOOLS_DIR, exist_ok=True)

def generate_tool_code(description, tool_name):
    """Ask Groq to generate a Python function."""
    prompt = f"""Write a Python function named '{tool_name}' that does the following: {description}
The function should take appropriate parameters and return a result. Do not include any extra text, only the function definition. Use proper indentation. Do not use external libraries unless absolutely necessary (prefer built-ins)."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        code = completion.choices[0].message.content
        # Basic validation: ensure it contains 'def'
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
    """Dynamically import a tool from the tools directory."""
    filepath = os.path.join(TOOLS_DIR, f"{tool_name}.py")
    if not os.path.exists(filepath):
        return None
    spec = importlib.util.spec_from_file_location(tool_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tool_name] = module
    spec.loader.exec_module(module)
    return getattr(module, tool_name, None)

@app.route('/create_tool', methods=['POST'])
def create_tool():
    data = request.get_json()
    if not data or 'name' not in data or 'description' not in data:
        return jsonify({"error": "Missing 'name' or 'description'"}), 400
    tool_name = data['name']
    description = data['description']
    
    # Generate code
    code, error = generate_tool_code(description, tool_name)
    if error:
        return jsonify({"error": f"Code generation failed: {error}"}), 500
    
    # Save tool
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
    args = data['args']  # should be a list of positional arguments
    
    tool_func = load_tool(tool_name)
    if not tool_func:
        return jsonify({"error": f"Tool '{tool_name}' not found"}), 404
    
    try:
        result = tool_func(*args)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Execution failed: {str(e)}", "traceback": traceback.format_exc()}), 500
