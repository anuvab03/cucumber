from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
from flask_cors import CORS
import os


base_path = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(base_path, 'templates'),
            static_folder=os.path.join(base_path, 'static'))
CORS(app)


model_path = os.path.join(base_path, 'model.h5')
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    raise


class_labels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6'}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = np.array(image)
    image = cv2.resize(image, (32, 32))  
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)  
    return image

def read_file_for_class(predicted_class):
    """Reads the corresponding text file based on the predicted class."""
    file_name = f"file_{predicted_class}.txt"  
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
