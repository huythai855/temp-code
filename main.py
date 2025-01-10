import os
import numpy as np
import json
import onnxruntime as ort
from flask import Flask, request, jsonify
from transformers import ElectraTokenizer

# Initialize Flask app
app = Flask(__name__)

# Path to the folder containing ONNX models
MODEL_DIR = "./model"

# Dictionary to store loaded ONNX sessions and tokenizers
model_sessions = {}
tokenizers = {}

# Helper function to load models dynamically
def load_models():
    for model_name in os.listdir(MODEL_DIR):
        if model_name.endswith(".onnx"):
            model_path = os.path.join(MODEL_DIR, model_name)
            # Load ONNX model session
            session = ort.InferenceSession(model_path)

            # Load tokenizer for Electra models (adjust if using other model types)
            if "electra" in model_name.lower():
                tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
                tokenizers[model_name] = tokenizer

            # Save session
            model_sessions[model_name] = session
            print(f"Loaded model: {model_name}")

# Load all models at startup
load_models()

# Dynamic route for each model
@app.route('/v1/inference/<model_name>', methods=['POST'])
def inference(model_name):
    # Check if model exists
    if model_name not in model_sessions:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    # Get input data
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. Provide 'text' in the JSON body."}), 400

    text = data['text']
    session = model_sessions[model_name]
    tokenizer = tokenizers.get(model_name)

    try:
        if tokenizer:
            # Tokenize input for Electra-based models
            inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
            input_feed = {
                session.get_inputs()[0].name: inputs["input_ids"].astype(np.int64),
                session.get_inputs()[1].name: inputs["attention_mask"].astype(np.float32),
            }
        else:
            # If no tokenizer is associated, assume raw input (adjust based on model requirements)
            input_feed = {
                session.get_inputs()[0].name: np.array([text]).astype(np.float32)  # Example for numeric input
            }

        # Run inference
        outputs = session.run(None, input_feed)
        print(outputs)

        # Chuyển từng phần tử trong outputs sang list nếu cần
        outputs_list = [output.tolist() if isinstance(output, np.ndarray) else output for output in outputs]

        return jsonify({"outputs": json.dumps(outputs_list)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to list all available models
@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({"models": list(model_sessions.keys())})

# Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
