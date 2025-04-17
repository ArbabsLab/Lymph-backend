from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io
import os
import requests

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/AteYourLunch/lymph-classifier/resolve/main/lymphmodel.pkl?download=true"
MODEL_PATH = "lymphmodel.pkl"


if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)


with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256)) 
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 256, 256, 3)
    return image_array

@app.route('/predict', methods=['POST'])
def predictLymph():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        input_data = preprocess_image(image_bytes)
        prediction = model.predict(input_data)
        predicted_class = int(np.argmax(prediction)) 
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
