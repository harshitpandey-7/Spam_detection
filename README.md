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

## Comparison

Run the comparison script to evaluate all models side by side:

```bash
python compare_all_models.py
```

This generates:
- `comparison_results.csv` - metrics table
- `comparison_bar_chart.png` - grouped bar chart
- `comparison_confusion_matrices.png` - confusion matrices for all models
- `comparison_radar_chart.png` - radar chart for top 5 models
- `comparison_f1_ranking.png` - F1-score ranking

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

- **Stemming vs Lemmatization** - two different text normalization approaches
- **Bag of Words vs TF-IDF** - raw word counts vs frequency-weighted features
- **Naive Bayes vs Logistic Regression vs SVM vs Random Forest** - different classification algorithms

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
