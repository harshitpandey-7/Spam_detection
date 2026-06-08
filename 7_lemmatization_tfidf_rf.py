# Spam Detection Model 7: Lemmatization + TF-IDF + Random Forest

import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load Dataset
messages = pd.read_csv('dataset/SMSSpamCollection', sep='\t', names=["label", "message"])
print(f"Dataset shape: {messages.shape}")
print(f"Label distribution:\n{messages['label'].value_counts()}\n")

# Data Cleaning and Preprocessing (Lemmatization)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    corpus.append(review)

# TF-IDF
tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("Model 7: Lemmatization + TF-IDF + Random Forest")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")

# Save results for comparison
results = {
    'model_name': 'Lemmatization + TF-IDF + RandomForest',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}

with open('results_7_lemma_tfidf_rf.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to results_7_lemma_tfidf_rf.pkl")
