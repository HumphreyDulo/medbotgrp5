import os
from flask import Flask, render_template, request, jsonify
import joblib
import re
import tensorflow as tf
import difflib
from spellchecker import SpellChecker



app = Flask(__name__)

# Load the trained model, vectorizer, and label encoder from disk
model_path = os.path.join('models', 'symptom_model.h5')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
label_encoder_path = os.path.join('models', 'label_encoder.pkl')

classifier = tf.keras.models.load_model(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

# Initialize the spell checker
spell = SpellChecker()

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


def correct_spelling(text):
    corrected_words = [spell.correction(word) for word in text.split()]
    return ' '.join(corrected_words)

def get_disease_from_question(question):
    # Correct spelling errors
    question = correct_spelling(question)

    # Clean and split the question for better matching
    question = clean_text(question)
    
    for disease in disease_symptoms.keys():
        if disease.lower() in question:
            return disease

    # Check for close matches if no exact match is found
    possible_matches = difflib.get_close_matches(question, disease_symptoms.keys(), n=1, cutoff=0.7)
    if possible_matches:
        return possible_matches[0]
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    symptom_description = request.json.get('symptoms')
    cleaned_text = clean_text(symptom_description)

    # Check if the user is asking for other symptoms of a specific disease
    if 'other symptoms' in cleaned_text:
        disease_name = cleaned_text.split('other symptoms of')[-1].strip()
        mentioned_symptoms = [symptom.strip() for symptom in re.split(r' and |,', symptom_description.lower())]
        other_symptoms = disease_symptoms.get(disease_name.title(), [])
        filtered_symptoms = [symptom for symptom in other_symptoms if symptom not in mentioned_symptoms]
        if filtered_symptoms:
            response = f"Other common symptoms of {disease_name.title()} include: {', '.join(filtered_symptoms)}."
        else:
            response = f"I don't have additional symptoms information for {disease_name.title()}."
        return jsonify({'response': response})

    # Check if the user is asking for symptoms of a specific disease
    if 'symptoms' in cleaned_text:
        disease_name = cleaned_text.split('symptoms of')[-1].strip()
        other_symptoms = disease_symptoms.get(disease_name.title(), [])
        if other_symptoms:
            response = f"{disease_name.title()} is typically associated with the following symptoms: {', '.join(other_symptoms)}."
        else:
            response = f"I don't have information about {disease_name.title()}."
        return jsonify({'response': response})

    # Check if the user input matches a known disease directly
    disease_name = cleaned_text.strip().title()
    disease = get_disease_from_question(disease_name)
    if disease:
        other_symptoms = disease_symptoms[disease]
        response = f"{disease} is typically associated with the following symptoms: {', '.join(other_symptoms)}."
        return jsonify({'response': response})

    # Predict the disease based on symptoms
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    predicted_label = classifier.predict(text_vector)
    predicted_label = label_encoder.inverse_transform([predicted_label.argmax()])[0]
    response = f'Based on the symptoms you provided, it is likely that you have {predicted_label}.'

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
