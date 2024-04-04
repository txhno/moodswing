from flask import Blueprint, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from model.preprocess import preprocess_text

main = Blueprint('main', __name__)

# Load the pre-trained model
model = load_model('trained_model/emotion_classifier.h5')

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    # Process input text and predict the corresponding emotion
    text = request.form['text']
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    # Assuming the output layer of your model corresponds to ['anger', 'joy', 'fear']
    emotions = ['anger', 'joy', 'fear']
    predicted_emotion = emotions[np.argmax(prediction)]
    return render_template('result.html', sentiment=predicted_emotion)
