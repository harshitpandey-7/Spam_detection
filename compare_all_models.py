# Compare All Spam Detection Models
# Runs all 7 pipelines and generates comparison charts

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# Load Dataset
messages = pd.read_csv('dataset/SMSSpamCollection', sep='\t', names=["label", "message"])
print(f"Dataset: {messages.shape[0]} messages")
print(f"Ham : {(messages['label']=='ham').sum()}")
print(f"Spam: {(messages['label']=='spam').sum()}\n")


# Preprocessing functions
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_stemming(texts):
    corpus = []
    for text in texts:
        review = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        review = [ps.stem(w) for w in review if w not in stop_words]
        corpus.append(' '.join(review))
    return corpus

def preprocess_lemmatization(texts):
    corpus = []
    for text in texts:
        review = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        review = [lemmatizer.lemmatize(w) for w in review if w not in stop_words]
        corpus.append(' '.join(review))
    return corpus


# Define all 7 model pipelines
pipelines = [
    {'name': '1. Stemming + BoW + NB',       'short': 'Stem+BoW+NB',    'preprocess': 'stemming',       'vectorizer': 'bow',   'classifier': MultinomialNB()},
    {'name': '2. Lemma + BoW + NB',           'short': 'Lem+BoW+NB',     'preprocess': 'lemmatization',  'vectorizer': 'bow',   'classifier': MultinomialNB()},
    {'name': '3. Stemming + TF-IDF + NB',     'short': 'Stem+TFIDF+NB',  'preprocess': 'stemming',       'vectorizer': 'tfidf', 'classifier': MultinomialNB()},
    {'name': '4. Lemma + TF-IDF + NB',        'short': 'Lem+TFIDF+NB',   'preprocess': 'lemmatization',  'vectorizer': 'tfidf', 'classifier': MultinomialNB()},
    {'name': '5. Lemma + TF-IDF + LR',        'short': 'Lem+TFIDF+LR',   'preprocess': 'lemmatization',  'vectorizer': 'tfidf', 'classifier': LogisticRegression(max_iter=1000, random_state=0)},
    {'name': '6. Lemma + TF-IDF + SVM',       'short': 'Lem+TFIDF+SVM',  'preprocess': 'lemmatization',  'vectorizer': 'tfidf', 'classifier': SVC(kernel='linear', random_state=0)},
    {'name': '7. Lemma + TF-IDF + RF',        'short': 'Lem+TFIDF+RF',   'preprocess': 'lemmatization',  'vectorizer': 'tfidf', 'classifier': RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)},
]

# Precompute the two preprocessed corpora
print("Preprocessing text...", end=" ")
corpus_stemmed = preprocess_stemming(messages['message'])
corpus_lemmatized = preprocess_lemmatization(messages['message'])
print("Done!\n")

y = pd.get_dummies(messages['label']).iloc[:, 1].values


# Run all pipelines
all_results = []
all_cm = []

for pipe in pipelines:
    print(f"Running: {pipe['name']}...", end=" ")

    corpus = corpus_stemmed if pipe['preprocess'] == 'stemming' else corpus_lemmatized

    if pipe['vectorizer'] == 'bow':
        vec = CountVectorizer(max_features=5000)
    else:
        vec = TfidfVectorizer(max_features=5000)

    X = vec.fit_transform(corpus).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    model = pipe['classifier']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    all_results.append({
        'Model': pipe['name'],
        'Short': pipe['short'],
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
    })
    all_cm.append(cm)
    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")


# Results Table
df = pd.DataFrame(all_results)

print("\n" + "=" * 80)
print("  COMPARISON TABLE")
print("=" * 80)
print(df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))


# Best model per metric
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
print("\n" + "=" * 80)
print("  BEST MODEL PER METRIC")
print("=" * 80)
for metric in metrics:
    best_idx = df[metric].idxmax()
    print(f"  {metric:10s} -> {df.loc[best_idx, 'Model']}  ({df.loc[best_idx, metric]:.4f})")

overall_best = df.loc[df['F1-Score'].idxmax()]
print(f"\n  OVERALL BEST (by F1-Score): {overall_best['Model']}  (F1={overall_best['F1-Score']:.4f})")


# Insights
print("\n" + "=" * 80)
print("  INSIGHTS")
print("=" * 80)

# Stemming vs Lemmatization (BoW + NB)
stem_bow_nb = df[df['Short'] == 'Stem+BoW+NB']['F1-Score'].values[0]
lem_bow_nb  = df[df['Short'] == 'Lem+BoW+NB']['F1-Score'].values[0]
print(f"\n  Stemming vs Lemmatization (BoW + NB):")
print(f"     Stemming F1 = {stem_bow_nb:.4f}  |  Lemmatization F1 = {lem_bow_nb:.4f}")

# Stemming vs Lemmatization (TF-IDF + NB)
stem_tfidf_nb = df[df['Short'] == 'Stem+TFIDF+NB']['F1-Score'].values[0]
lem_tfidf_nb  = df[df['Short'] == 'Lem+TFIDF+NB']['F1-Score'].values[0]
print(f"\n  Stemming vs Lemmatization (TF-IDF + NB):")
print(f"     Stemming F1 = {stem_tfidf_nb:.4f}  |  Lemmatization F1 = {lem_tfidf_nb:.4f}")

# BoW vs TF-IDF (Lemma + NB)
print(f"\n  BoW vs TF-IDF (Lemma + NB):")
print(f"     BoW F1 = {lem_bow_nb:.4f}  |  TF-IDF F1 = {lem_tfidf_nb:.4f}")

# Classifier comparison (all Lemma + TF-IDF)
lem_tfidf_lr  = df[df['Short'] == 'Lem+TFIDF+LR']['F1-Score'].values[0]
lem_tfidf_svm = df[df['Short'] == 'Lem+TFIDF+SVM']['F1-Score'].values[0]
lem_tfidf_rf  = df[df['Short'] == 'Lem+TFIDF+RF']['F1-Score'].values[0]
print(f"\n  Classifier Comparison (all Lemma + TF-IDF):")
print(f"     NB  F1 = {lem_tfidf_nb:.4f}")
print(f"     LR  F1 = {lem_tfidf_lr:.4f}")
print(f"     SVM F1 = {lem_tfidf_svm:.4f}")
print(f"     RF  F1 = {lem_tfidf_rf:.4f}")
clf_scores = {'NB': lem_tfidf_nb, 'LR': lem_tfidf_lr, 'SVM': lem_tfidf_svm, 'RF': lem_tfidf_rf}
best_clf = max(clf_scores, key=clf_scores.get)
print(f"     Best classifier: {best_clf} (F1={clf_scores[best_clf]:.4f})")


# Generating plots
print("\nGenerating plots...", end=" ")

# Plot 1: Grouped Bar Chart
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(df))
width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i * width, df[metric], width, label=metric, color=colors[i], edgecolor='white')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Spam Detection - All Models Comparison', fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df['Short'], rotation=30, ha='right', fontsize=9)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.7, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Confusion Matrix Heatmaps
rows = 2
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
axes = axes.flatten()

for idx, (result, cm) in enumerate(zip(all_results, all_cm)):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                cbar=False, annot_kws={'size': 14})
    ax.set_title(result['Short'], fontsize=11)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)

# hide the extra subplot (we have 7 models but 2x4=8 cells)
axes[7].set_visible(False)

plt.suptitle('Confusion Matrices - All Models', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('comparison_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Radar Chart for top 5 models
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

top5 = df.nlargest(5, 'F1-Score')
colors_radar = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

for i, (_, row) in enumerate(top5.iterrows()):
    values = [row[m] for m in metrics]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=row['Short'], color=colors_radar[i])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.7, 1.05)
ax.set_title('Top 5 Models - Radar Chart', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig('comparison_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: F1-Score Ranking
fig, ax = plt.subplots(figsize=(10, 6))
df_sorted = df.sort_values('F1-Score', ascending=True)
colors_rank = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_sorted)))

bars = ax.barh(df_sorted['Short'], df_sorted['F1-Score'], color=colors_rank, edgecolor='white', height=0.6)
for bar, val in zip(bars, df_sorted['F1-Score']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=10)

ax.set_xlabel('F1-Score', fontsize=12)
ax.set_title('F1-Score Ranking - All Models', fontsize=14)
ax.set_xlim(0.7, 1.05)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_f1_ranking.png', dpi=150, bbox_inches='tight')
plt.close()

print("Done!")

# Save comparison table to CSV
df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_csv('comparison_results.csv', index=False)

print("\nFiles saved:")
print("  comparison_results.csv")
print("  comparison_bar_chart.png")
print("  comparison_confusion_matrices.png")
print("  comparison_radar_chart.png")
print("  comparison_f1_ranking.png")
