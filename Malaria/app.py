import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
app.config['UPLOAD_FOLDER'] = os.path.join('backend', 'static', 'uploads')

# Load the trained model for malaria detection
model = load_model('saved_model.h5')

# Load malaria information from JSON file
with open('malaria_info.json', 'r') as f:
    malaria_info = json.load(f)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define a function to process the uploaded image
def process_image(image):
    image = cv2.resize(image, (180, 180))  # Resize image to match model input size
    image = image / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)
    return image

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
@cross_origin(origin='http://localhost:3000', headers=['Content-Type'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 500

        processed_img = process_image(img)
        pred = model.predict(processed_img)
        predicted_class = np.argmax(pred[0])  # Assuming you have a classification model

        # Mapping from model prediction to malaria type
        malaria_types = ["Malariae", "Ovale", "Vivax", "Falciparum"]
        result = malaria_types[predicted_class]

        severity = malaria_info[result]['severity']
        info = malaria_info[result]['info']
        symptoms = malaria_info[result]['symptoms']
        treatment = malaria_info[result]['treatment']
        geography = malaria_info[result]['geography']
        prevention = malaria_info[result]['prevention']

        return jsonify({
            'result': result,
            'severity': severity,
            'info': info,
            'symptoms': symptoms,
            'treatment': treatment,
            'geography': geography,
            'prevention': prevention
        })

    return jsonify({'error': 'Failed to process the image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
