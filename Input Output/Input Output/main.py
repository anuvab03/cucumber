from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
from flask_cors import CORS
import os

# Create the base path handling spaces in folder name
base_path = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(base_path, 'templates'),
            static_folder=os.path.join(base_path, 'static'))
CORS(app)

# Load model with proper path handling
model_path = os.path.join(base_path, 'model.h5')
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Class labels mapped to numbers 1-6
class_labels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6'}

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (32, 32))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def read_file_for_class(predicted_class):
    """Reads a file based on the predicted class number (1-6)."""
    file_name = f"file_{predicted_class}.txt"  # Example: file_1.txt, file_2.txt, etc.
    file_path = os.path.join(base_path, 'files', file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "No relevant file found."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)
    
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)

    # Ensure the prediction maps to 1-6
    if predicted_class_index in class_labels:
        predicted_class = class_labels[predicted_class_index]
        file_content = read_file_for_class(predicted_class)
    else:
        return jsonify({'error': 'Unexpected prediction class'}), 500

    return jsonify({
        'prediction': predicted_class,
        'confidence': f"{float(np.max(prediction)) * 100:.2f}%",
        'file_content': file_content
    })

if __name__ == '__main__':
    app.run(debug=True)
