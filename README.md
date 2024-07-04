

---

# Symptom Checker Chatbot

## Overview

MedBot is an NLP-based chatbot designed to predict possible diseases based on user-provided symptoms. The chatbot uses machine learning models to analyze symptoms and provide likely diagnoses. It is built with Flask for the backend and a user-friendly web interface.

## Features

- **Symptom Analysis**: Predicts possible diseases based on input symptoms.
- **Machine Learning**: Utilizes trained models for accurate disease prediction.
- **User-Friendly Interface**: Simple and intuitive web-based chatbot interface.

## Setup Instructions

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- scikit-learn
- Joblib
- Pandas

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/HumphreyDulo/medbotgrp5.git
    cd medbotgrp5
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
   pip install pandas scikit-learn tensorflow joblib Flask
    ```

4. **Download Dataset**:
    Place your dataset (`Symptom2Disease.csv`) in the `datasets` directory.

### Running the Application

1. **Train the Model**:
    ```bash
    python main.py
    ```

2. **Run the Application**:
    ```bash
    python app.py
    ```

3. **Access the Chatbot**:
    Open your browser and navigate to `http://127.0.0.1:5000/`.

4. **Performance Testing**:
    To compare the performance of the other models, run:
    ```bash
    python test.py
    ```

## Project Structure

- **main.py**: Script to train the machine learning model.
- **app.py**: Main Flask application file to run the chatbot.
- **test.py**: Script for performance measurement and comparison of the other models.
- **models/**: Directory containing machine learning models (`symptom_model.h5`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`).
- **templates/**: HTML templates for the web interface.
- **datasets/**: Directory for storing datasets.



## Contributors
- 142285 Sumeiya Ali
- 145351 Kihanya Mungai
- 146424 Masoud Hassan
- 145835 Humphrey James

---
