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

# Disease symptoms dictionary
disease_symptoms = {
    "Psoriasis": ["red patches of skin", "dry skin", "itching"],
    "Varicose Veins": ["veins that appear twisted and bulging", "pain and discomfort", "swelling in legs"],
    "Typhoid": ["high fever", "weakness", "stomach pain"],
    "Chicken pox": ["itchy red rash", "fever", "fatigue"],
    "Impetigo": ["red sores", "blisters", "itching"],
    "Dengue": ["high fever", "severe headache", "pain behind the eyes"],
    "Fungal infection": ["itching", "redness", "skin peeling"],
    "Common Cold": ["sneezing", "runny nose", "sore throat"],
    "Pneumonia": ["cough with phlegm", "fever", "shortness of breath"],
    "Dimorphic Hemorrhoids": ["painful bowel movements", "itching in rectal area", "bleeding"],
    "Arthritis": ["joint pain", "stiffness", "swelling"],
    "Acne": ["pimples", "blackheads", "oily skin"],
    "Bronchial Asthma": ["shortness of breath", "chest tightness", "wheezing"],
    "Hypertension": ["high blood pressure", "headaches", "shortness of breath"],
    "Migraine": ["intense headache", "nausea", "sensitivity to light"],
    "Cervical spondylosis": ["neck pain", "stiffness", "headache"],
    "Jaundice": ["yellowing of the skin", "fatigue", "dark urine"],
    "Malaria": ["fever", "chills", "sweating"],
    "Urinary tract infection": ["burning sensation when urinating", "frequent urination", "cloudy urine"],
    "Allergy": ["sneezing", "itchy eyes", "runny nose"],
    "Gastroesophageal reflux disease": ["heartburn", "regurgitation", "difficulty swallowing"],
    "Drug reaction": ["rash", "itching", "swelling"],
    "Peptic ulcer disease": ["stomach pain", "bloating", "heartburn"],
    "Diabetes": ["increased thirst", "frequent urination", "fatigue"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_disease_from_question(question):
    for disease in disease_symptoms.keys():
        if disease.lower() in question.lower():
            return disease
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptom_description = request.json.get('symptoms')
    cleaned_text = clean_text(symptom_description)

    # Check if the user is asking for symptoms of a specific disease
    disease_from_question = get_disease_from_question(cleaned_text)
    if disease_from_question:
        other_symptoms = disease_symptoms.get(disease_from_question, [])
        response = f"Possible common symptoms of {disease_from_question} include: {', '.join(other_symptoms)}."
        return jsonify({'response': response})

    text_vector = vectorizer.transform([cleaned_text]).toarray()
    predicted_label = classifier.predict(text_vector)
    predicted_label = label_encoder.inverse_transform([predicted_label.argmax()])[0]
    
    other_symptoms = disease_symptoms.get(predicted_label, [])
    response = f"Based on the symptoms you provided, it is likely that you have {predicted_label}. "
    if other_symptoms:
        response += f"Other common symptoms of {predicted_label} include: {', '.join(other_symptoms)}."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
