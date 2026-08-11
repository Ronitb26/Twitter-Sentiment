# Twitter US Airline Sentiment Analysis

Classifies airline-related tweets as **negative / neutral / positive**, benchmarking a tuned TF-IDF +
Linear SVM baseline against SimpleRNN, LSTM, and Bidirectional LSTM models — all evaluated on the same
held-out test set.

## Headline result (reproduced, not estimated)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Linear SVM — tuned (TF-IDF)** | **79.8%** | **0.735** |
| Bidirectional LSTM | 78.4% | 0.722 |
| Logistic Regression (TF-IDF) | 77.2% | 0.725 |
| LSTM | 75.6% | 0.645 |
| SimpleRNN | 63.6% | 0.437 |

**The tuned linear baseline beats every neural model tried, including a Bidirectional LSTM.** That's the
actual finding of this project, not a caveat — see the notebook's conclusion for why, and what that implies
about when deep learning earns its cost.

## Problem

Airlines get thousands of tweets a day. Manually triaging them for customer-service escalation doesn't
scale. This project builds an automated sentiment classifier on a public dataset of ~14.6K tweets directed
at 6 US airlines (Kaggle "Twitter US Airline Sentiment"), treating the 63/21/16 class imbalance
(negative/neutral/positive) as a first-class constraint rather than an afterthought.

## Approach

1. **Data leakage fix** — the original approach built its vocabulary (word counts, one-hot hashing) on the
   full dataset *before* splitting train/test. This version splits first and fits every vectorizer /
   tokenizer only on the training set.
2. **Preprocessing bug fixes** — two silent bugs caught by inspecting actual output instead of trusting the
   code: a mention-removal regex that required a space after `@` (so `@VirginAmerica` never matched), and a
   stopword filter whose boolean logic made it a near no-op due to operator precedence.
3. **Class imbalance handling** — `class_weight='balanced'` for the linear models, computed class weights
   for the neural models, and **macro F1** reported alongside accuracy throughout (a model predicting
   "negative" for everything would already hit ~63% accuracy, so accuracy alone isn't trustworthy here).
4. **Real model comparison, not just one model** — TF-IDF + Logistic Regression, a tuned TF-IDF + Linear
   SVM (small grid search over `C`), SimpleRNN, LSTM, and Bidirectional LSTM are all trained and evaluated
   on the identical held-out test set with classification reports and confusion matrices.
5. **Honest conclusion** — the notebook doesn't default to "deep learning wins." It reports which model
   actually won on macro F1 and reasons about *why* (dataset size, tweet length, ease of regularizing a
   convex model) before recommending the model to ship.

## Stack

`pandas` · `scikit-learn` (TF-IDF, Logistic Regression, LinearSVC, metrics) · `TensorFlow/Keras`
(Tokenizer, Embedding, SimpleRNN, LSTM, Bidirectional LSTM) · `nltk` (tokenization, stopwords) ·
`matplotlib` / `seaborn`

## Repo contents

- `twitter_sentiment_analysis_optimized.ipynb` — full notebook: EDA → preprocessing → 2 classical baselines → SimpleRNN → LSTM → BiLSTM → comparison
- `Tweets.csv` — dataset
- `interview_prep.md` — STAR narrative, likely interview questions, and resume bullets
