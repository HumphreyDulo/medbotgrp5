import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
file_path = 'datasets/Symptom2Disease.csv'
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
tfidf_matrix_dense = tfidf_matrix.toarray()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix_dense, df['label_encoded'], test_size=0.2, random_state=42)

# Define models to compare (Decision Tree and Naive Bayes)
models = [
    DecisionTreeClassifier(random_state=42),
    MultinomialNB(),
]

# Train and evaluate each model
for model in models:
    model_name = type(model).__name__
    print(f"Training and evaluating {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(f"{model_name} Classification Report:")
    print(report)
