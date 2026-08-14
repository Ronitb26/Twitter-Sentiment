# Twitter Sentiment Analysis

Classifying tweets as **negative / neutral / positive**, comparing two feature-extraction strategies (TF-IDF vs. pretrained word embeddings) across two families of models (classical ML vs. deep learning).

## Dataset

- **~14,640 tweets** directed at 6 major US airlines.
- Each tweet is labeled `negative`, `neutral`, or `positive`.
- **Class distribution is heavily imbalanced:** ~63% negative, 21% neutral, 16% positive.

## Approach

### 1. EDA
- Visualized sentiment distribution overall and broken down by airline.
- Noted that sentiment skew varies by airline, but deliberately **excluded airline as a feature** — in a real deployment, this model would score incoming tweets about a *new* or *unseen* airline, and baking in airline-specific bias would hurt generalization and could unfairly penalize/favor specific airlines.

### 2. Text Preprocessing
- Lowercased text, stripped `@mentions` and URLs, removed non-alphabetic characters.
- Tokenized with NLTK and removed stopwords — **except a negation whitelist** (`not`, `no`, `never`, `n't`) which is deliberately kept, since dropping negations flips sentiment meaning (e.g. "not good" → "good" would be a labeling disaster).

### 3. Train/Test Split
- Stratified 80/20 split (11,712 train / 2,928 test) to preserve class ratios in both sets.

### 4. Feature Extraction — two strategies compared
| Strategy | Used by |
|---|---|
| **TF-IDF** (unigrams + bigrams, up to 30K features) | Logistic Regression, Complement Naive Bayes, Linear SVM |
| **Pretrained GloVe-Twitter-100 embeddings** (86.1% vocab coverage) | LSTM, BiLSTM |

### 5. Models Trained
- **Logistic Regression** (`class_weight='balanced'`) — linear baseline.
- **Complement Naive Bayes** — a Naive Bayes variant designed specifically for imbalanced text classification.
- **Linear SVM** — tuned via a small grid search over `C ∈ {0.1, 0.3, 1.0, 3.0}`.
- **LSTM** — single-direction, 2-layer, with dropout + recurrent dropout + batch norm, trained on padded sequences initialized with GloVe embeddings.
- **Bidirectional LSTM** — same idea, reads sequences in both directions to capture context from both sides of a word.
- Both neural models use **computed class weights**, **early stopping** on validation loss, and **softmax + categorical cross-entropy** for the 3-class output.

### 6. Evaluation
- **Accuracy** and **macro-F1** are both tracked — macro-F1 is the primary metric of interest because of the class imbalance (it weights all three classes equally, so the model can't hide poor minority-class performance behind majority-class accuracy).
- Confusion matrices generated for the best classical model (tuned Linear SVM) and the best neural model (BiLSTM).

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **BiLSTM** | 0.7763 | **0.7353** |
| **Linear SVM (tuned, C=0.1)** | **0.7978** | 0.7348 |
| LSTM | 0.7790 | 0.7308 |
| Logistic Regression | 0.7719 | 0.7253 |
| Complement NB | 0.7787 | 0.7018 |

## Key Insight

A well-tuned **linear SVM on TF-IDF features essentially matches a BiLSTM with pretrained embeddings** — the deep learning model wins narrowly on macro-F1 (+0.0005) while the SVM wins on raw accuracy, at a fraction of the training time and compute. On a dataset of this size (~15K short, noisy tweets), the added representational power of sequence models doesn't translate into a clear win.

## Tech Stack
`Python` · `pandas` / `numpy` · `scikit-learn` · `NLTK` · `TensorFlow / Keras` · `Gensim` (GloVe-Twitter-100) · `matplotlib` / `seaborn`

## Project Structure
```
.
├── twitter_sentiment_analysis.ipynb   # full pipeline: EDA → preprocessing → TF-IDF/classical models → embeddings → LSTM/BiLSTM → comparison
├── Tweets.csv                          # dataset (US Airline Twitter Sentiment)
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt   # pandas, scikit-learn, nltk, tensorflow, gensim, seaborn, matplotlib
jupyter notebook twitter_sentiment_analysis.ipynb
```
