import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib

# Load your dataset
file_path = 'datasets/Symptom2Disease.csv'  # Adjust the path relative to your project directory
df = pd.read_csv(file_path)

# Check for NaN and infinite values in numerical columns
print(df.isnull().sum())

# Drop rows with NaN values
df.dropna(subset=['label', 'text'], inplace=True)

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Vectorize the symptom text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Convert tfidf_matrix to dense numpy array
tfidf_matrix_dense = tfidf_matrix.toarray()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix_dense, df['label_encoded'], test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate with classification report
y_pred_rf = model_rf.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_)
print("Random Forest Classification Report:")
print(report_rf)

# Convert RandomForestClassifier to Keras Sequential model (for .h5 deployment)
model_keras = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(tfidf_matrix_dense.shape[1],)),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the Keras model
model_keras.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Fit the Keras model using dense numpy array
model_keras.fit(tfidf_matrix_dense, df['label_encoded'], epochs=10, validation_split=0.2)

# Save the Keras model as .h5 file
model_keras.save('models/symptom_model.h5')

# Save the TF-IDF vectorizer and label encoder
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
