# Twitter Sentiment Analysis

Classifying tweets about specific entities (games, tech brands, companies) as **Negative / Neutral / Positive**, comparing three modeling approaches: **Naive Bayes**, **Random Forest**, and a **Bidirectional LSTM** with a trainable embedding.

## Dataset

- **~74.7K labeled tweets** across **32 entities** (games, tech companies, brands) — used as the training pool.
- A **separate, independently-collected 1K-tweet validation set**, kept fully untouched until final evaluation.
- 3 classes after label consolidation: ~42% Neutral, 31% Negative, 27% Positive.

### On the label scheme

The source dataset ships a 4th class, `Irrelevant`, for tweets that mention an entity without expressing real sentiment. It's merged into `Neutral` here to keep the problem to a straightforward 3-class polarity task. It's a genuine simplification of the task, not evidence the models themselves will get better.

## Approach

### 1. Cleaning
- Dropped 686 rows with missing text and ~4,900 duplicate (text, sentiment) pairs from training data (mostly near-identical paraphrased tweets in the raw data).

### 2. Preprocessing — spaCy lemmatization
- Light regex pass first (lowercase, strip `@mentions`/URLs).
- spaCy (`en_core_web_sm`, with `parser`/`ner` disabled for speed) handles tokenization, POS-aware **lemmatization**, and stopword/punctuation removal.
- Stopword removal **excludes a negation whitelist** (`not`, `no`, `never`, `n't`) — dropping negations would collapse "not good" into "good," destroying the actual sentiment signal.

### 3. Feature Extraction & Models
| Model | Features | Imbalance handling |
|---|---|---|
| **Naive Bayes** | TF-IDF (unigrams+bigrams, 8K features), `sklearn.Pipeline` | `fit_prior=False` |
| **Random Forest** | TF-IDF (unigrams+bigrams, 8K features), `sklearn.Pipeline` | `class_weight='balanced'` |
| **Bidirectional LSTM** | Trainable embedding over tokenized/padded sequences | `class_weight` in the loss function |


### 4. Evaluation
- Accuracy and macro-F1 tracked for all three models, evaluated on the untouched validation set.
- Confusion matrices generated for each.
- A single-tweet inference demo at the end shows the shipped pipeline being used exactly as it would in production.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Bidirectional LSTM** | **0.9379** | **0.9352** |
| Random Forest | 0.7928 | 0.7919 |
| Naive Bayes | 0.7497 | 0.7505 |

## Key Insight

The **Bidirectional LSTM decisively outperforms both classical models**, consistent with a companion sentiment project's finding that deep sequence models pull ahead when there's enough training data and the task involves context-dependent slang (e.g. "murder"/"kill" as gaming hype-speech, not literal violence) — signal a bag-of-words TF-IDF representation mostly can't capture. Between the classical models, **Random Forest edges out Naive Bayes**, as expected: Naive Bayes' conditional-independence assumption between words is a stronger simplification than Random Forest's ability to model feature interactions.

## Tech Stack
`Python` · `pandas` / `numpy` · `scikit-learn` · `spaCy` · `TensorFlow / Keras` · `matplotlib` / `seaborn`

## Project Structure
```
.
├── Twitter_Sentiment_Analysis.ipynb   # full pipeline: cleaning → label merge → spaCy preprocessing → TF-IDF/NB/RF → tokenizing → BiLSTM → comparison 
├── twitter_training.csv                       # training pool (~74.7K tweets)
├── twitter_validation.csv                     # untouched validation set (1K tweets)
└── README.md
```
