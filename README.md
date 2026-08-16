# Twitter Entity Sentiment Analysis

Classifying tweets about specific entities (games, tech brands, companies) as **Negative / Neutral / Positive**, comparing three modeling approaches: **Naive Bayes**, **Random Forest**, and a **Bidirectional LSTM** with a trainable embedding.

## Dataset

- **~74.7K labeled tweets** across **32 entities** (games, tech companies, brands) — used as the training pool.
- A **separate, independently-collected 1K-tweet validation set**, kept fully untouched until final evaluation.
- 3 classes after label consolidation: ~42% Neutral, 31% Negative, 27% Positive.

### On the label scheme

The source dataset ships a 4th class, `Irrelevant`, for tweets that mention an entity without expressing real sentiment. It's merged into `Neutral` here to keep the problem to a straightforward 3-class polarity task. This is a **deliberate scope decision, not a modeling improvement** — `Irrelevant` was consistently the hardest class to separate in a 4-class version of this project (lowest precision across every model), so merging it raises the reported metrics. Worth being upfront about: it's a genuine simplification of the task, not evidence the models themselves got better.

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

**Why no SMOTE:** this dataset's imbalance is modest (~42/31/27%), and SMOTE's synthetic oversampling adds a "does this fabricated data represent reality" defensibility question that's hard to justify for an imbalance this mild. **Class weighting** — reweighting the loss on real data instead of manufacturing synthetic points — is used consistently across all three models instead.

**Why a trainable embedding over pretrained GloVe:** both were tested empirically during development. Pretrained GloVe-Twitter-100 only covered **76.8%** of the training vocabulary (this dataset's gaming/tech slang isn't well represented in generic pretrained vectors) and scored 0.9048 macro-F1. A trainable embedding, learned end-to-end on this dataset's ~62K training sequences, scored **0.9252 macro-F1** — clearly ahead. With enough in-domain data, task-specific embeddings learned from scratch outperformed importing general-purpose ones.

**Why sklearn `Pipeline`:** both classical models bundle the TF-IDF vectorizer and classifier into a single object — a small production-readiness choice, since the fitted pipeline can be pickled and shipped as one artifact and called directly on raw cleaned text at inference time.

### 4. Evaluation
- Accuracy and macro-F1 tracked for all three models, evaluated on the untouched validation set.
- Confusion matrices generated for each.
- A single-tweet inference demo at the end shows the shipped pipeline being used exactly as it would in production.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Bidirectional LSTM** | **0.9279** | **0.9252** |
| Random Forest | 0.7938 | 0.7929 |
| Naive Bayes | 0.7497 | 0.7505 |

## Key Insight

The **Bidirectional LSTM decisively outperforms both classical models**, consistent with a companion sentiment project's finding that deep sequence models pull ahead when there's enough training data and the task involves context-dependent slang (e.g. "murder"/"kill" as gaming hype-speech, not literal violence) — signal a bag-of-words TF-IDF representation mostly can't capture. Between the classical models, **Random Forest edges out Naive Bayes**, as expected: Naive Bayes' conditional-independence assumption between words is a stronger simplification than Random Forest's ability to model feature interactions.

Every major methodology choice in this project — imbalance strategy, embedding type, label scope — was tested empirically rather than assumed, with each decision backed by a specific, reproducible number.

## Tech Stack
`Python` · `pandas` / `numpy` · `scikit-learn` · `spaCy` · `TensorFlow / Keras` · `matplotlib` / `seaborn`

## Project Structure
```
.
├── Twitter_Entity_Sentiment_Analysis.ipynb   # full pipeline: cleaning → label merge → spaCy preprocessing → TF-IDF/NB/RF → tokenizing → BiLSTM → comparison → demo
├── twitter_training.csv                       # training pool (~74.7K tweets)
├── twitter_validation.csv                     # untouched validation set (1K tweets)
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt   # pandas, scikit-learn, spacy, tensorflow, seaborn, matplotlib
python -m spacy download en_core_web_sm
jupyter notebook Twitter_Entity_Sentiment_Analysis.ipynb
```

## Possible Extensions
- Fine-tune a transformer (e.g. DistilBERT) as an upper-bound comparison point.
- Per-entity error analysis — check whether the model struggles more on certain entities (e.g. more sarcastic gaming communities) than others.
- A lightweight Streamlit/Gradio demo wrapping the saved Random Forest pipeline for a live, shareable inference demo.
