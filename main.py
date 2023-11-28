from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16

app = Flask(__name__)

# Load your trained model
model = load_model('defects_model1.h5')  # Update with your model path

def preprocess_image(image_path, label):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to the desired size
    # Preprocess the image for VGG16
    image = image.astype(np.float32)
    image = image[..., ::-1]  # BGR to RGB
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    return image, label

@app.route('/')
def home():
    return 'index.html'

@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part'})

    file = request.files['file']
    print(file)

    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    print(file_path)

    # Preprocess the uploaded image
    input_image, _ = preprocess_image(file_path, None)

# Feature extraction using VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    input_features = vgg16_model.predict(np.expand_dims(input_image, axis=0))

# Flatten the features
    input_features_flat = input_features.reshape((input_features.shape[0], -1))

# Make a prediction
    prediction = model.predict(input_features_flat)

    # # Make a prediction
    # print("input_data",input_data)
    # prediction = model.predict(input_data)

    # Return the prediction as JSON
    modi_prediction=float(prediction[0][0])
    if modi_prediction>0.5:
        return jsonify({'result':'Non-Defective'})
    else:
        return jsonify({'result':'Defective'})
    

if __name__ == '__main__':
    app.run(port=os.getenv('PORT', default=5001))

