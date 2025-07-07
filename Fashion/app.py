import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import uuid

app = Flask(__name__)
model = load_model("fashion_model.hdf5")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded."

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    
    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]

    return render_template('result.html', predicted_class=predicted_label, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
