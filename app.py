import os
from flask import Flask, render_template, request, jsonify
import joblib
import re
import tensorflow as tf

app = Flask(__name__)

# Load the trained model, vectorizer, and label encoder from disk
model_path = os.path.join('models', 'symptom_model.h5')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
label_encoder_path = os.path.join('models', 'label_encoder.pkl')

classifier = tf.keras.models.load_model(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    symptom_description = request.json.get('symptoms')
    cleaned_text = clean_text(symptom_description)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    predicted_label = classifier.predict(text_vector)
    predicted_label = label_encoder.inverse_transform([predicted_label.argmax()])[0]
    return jsonify({'response': f'Based on the symptoms you provided, it is likely that you have {predicted_label}.'})


if __name__ == '__main__':
    app.run(debug=True)
