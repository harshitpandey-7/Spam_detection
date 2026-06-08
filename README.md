# SMS Spam Detection - Model Comparison

Comparing different text preprocessing and classification techniques for SMS spam detection using the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

## Dataset

The dataset contains 5,572 SMS messages labeled as `ham` (legitimate) or `spam`. It is stored in `dataset/SMSSpamCollection`.

## Models

| # | File | Preprocessing | Vectorizer | Classifier |
|---|------|---------------|------------|------------|
| 1 | `1_stemming_bow_nb.py` | Porter Stemming | Bag of Words | Multinomial Naive Bayes |
| 2 | `2_lemmatization_bow_nb.py` | WordNet Lemmatization | Bag of Words | Multinomial Naive Bayes |
| 3 | `3_stemming_tfidf_nb.py` | Porter Stemming | TF-IDF | Multinomial Naive Bayes |
| 4 | `4_lemmatization_tfidf_nb.py` | WordNet Lemmatization | TF-IDF | Multinomial Naive Bayes |
| 5 | `5_lemmatization_tfidf_lr.py` | WordNet Lemmatization | TF-IDF | Logistic Regression |
| 6 | `6_lemmatization_tfidf_svm.py` | WordNet Lemmatization | TF-IDF | SVM (Linear) |
| 7 | `7_lemmatization_tfidf_rf.py` | WordNet Lemmatization | TF-IDF | Random Forest |

All models use `max_features=5000`.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| 1. Stemming + BoW + NB | 0.9821 | 0.9268 | **0.9500** | **0.9383** |
| 2. Lemma + BoW + NB | 0.9821 | 0.9268 | **0.9500** | **0.9383** |
| 3. Stemming + TF-IDF + NB | 0.9740 | **1.0000** | 0.8188 | 0.9003 |
| 4. Lemma + TF-IDF + NB | 0.9767 | **1.0000** | 0.8375 | 0.9116 |
| 5. Lemma + TF-IDF + LR | 0.9641 | 0.9839 | 0.7625 | 0.8592 |
| 6. Lemma + TF-IDF + SVM | **0.9830** | 0.9930 | 0.8875 | 0.9373 |
| 7. Lemma + TF-IDF + RF | 0.9812 | **1.0000** | 0.8688 | 0.9298 |

### Best Model Per Metric

| Metric | Best Model | Score |
|--------|-----------|-------|
| Accuracy | Lemma + TF-IDF + SVM | 0.9830 |
| Precision | Stemming + TF-IDF + NB / Lemma + TF-IDF + RF | 1.0000 |
| Recall | Stemming + BoW + NB / Lemma + BoW + NB | 0.9500 |
| F1-Score | Stemming + BoW + NB / Lemma + BoW + NB | 0.9383 |

### Comparison Charts

#### All Models Comparison
![Bar Chart](comparison_bar_chart.png)

#### F1-Score Ranking
![F1 Ranking](comparison_f1_ranking.png)

#### Confusion Matrices
![Confusion Matrices](comparison_confusion_matrices.png)

#### Top 5 Models Radar Chart
![Radar Chart](comparison_radar_chart.png)

## Conclusion

### 1. Stemming vs Lemmatization

With BoW + Naive Bayes both give identical results (F1 = 0.9383). With TF-IDF + NB, lemmatization slightly outperforms stemming (F1: 0.9116 vs 0.9003).

**Why?** Stemming chops words aggressively (e.g. "running", "runner", "runs" all become "run"), which reduces vocabulary size and groups more words together. This works well with BoW where raw frequency matters. Lemmatization is more conservative — it only reduces words to their dictionary form (e.g. "running" becomes "running", "ran" becomes "run"), so it preserves more semantic information. With TF-IDF, where word importance is weighted, this extra precision from lemmatization helps slightly.

### 2. Bag of Words vs TF-IDF

BoW consistently outperforms TF-IDF when paired with Naive Bayes (F1: 0.9383 vs 0.9116).

**Why?** Multinomial Naive Bayes works on raw event counts — it calculates the probability of each word appearing in spam vs ham. BoW gives raw counts which is exactly what NB expects. TF-IDF normalizes and down-weights frequent words, which actually removes information that NB could use. The IDF component penalizes words that appear in many documents, but for spam detection, some common spam words (like "free", "win", "call") appear frequently and are strong indicators. TF-IDF suppresses them.

### 3. Classifier Comparison (all using Lemmatization + TF-IDF)

| Classifier | F1-Score | Why |
|------------|----------|-----|
| SVM | 0.9373 | Best with TF-IDF. SVM finds the maximum-margin hyperplane in high-dimensional space. TF-IDF features are normalized, which is what SVM prefers. Linear SVM is well suited for text classification. |
| RF | 0.9298 | Good overall. Random Forest handles high-dimensional sparse data well through feature bagging, but 100 trees may not fully capture the patterns. |
| NB | 0.9116 | Decent but lower recall. NB assumes feature independence, which is violated in text. With TF-IDF's normalized values, NB loses the count-based advantage it has with BoW. |
| LR | 0.8592 | Lowest F1. Logistic Regression struggles with the class imbalance (87% ham, 13% spam). Without tuning (class weights, regularization), it tends to predict the majority class. |

### 4. Why Recall Drops with TF-IDF

All TF-IDF models show lower recall (0.76-0.89) compared to BoW models (0.95). This means TF-IDF models miss more spam messages.

**Why?** TF-IDF down-weights words that appear across many documents. Some spam-indicator words like "call", "free", "text" also appear in legitimate messages, so their TF-IDF weight gets reduced. The model then fails to flag some spam messages that contain these shared words.

### 5. Why BoW + NB is the Best Overall

The simplest model wins because:
- The SMS dataset has short texts (average ~15 words), so word frequency itself is a strong signal
- The vocabulary is small enough that raw counts capture the patterns well
- Naive Bayes works best with integer counts (BoW) rather than float weights (TF-IDF)
- Stemming/Lemmatization reduces vocabulary, making the feature space manageable even at 5000 features

This is a known pattern in NLP — simpler models often outperform complex ones on small, clean datasets. The more advanced approaches (SVM, RF) would likely surpass NB on larger, noisier datasets.

## How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn
   ```
3. Run any individual model:
   ```bash
   python 1_stemming_bow_nb.py
   ```
4. Or run the full comparison:
   ```bash
   python compare_all_models.py
   ```

## What's Being Compared

- **Stemming vs Lemmatization** — two different text normalization approaches
- **Bag of Words vs TF-IDF** — raw word counts vs frequency-weighted features
- **Naive Bayes vs Logistic Regression vs SVM vs Random Forest** — different classification algorithms

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
