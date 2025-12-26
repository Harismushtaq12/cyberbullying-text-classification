# Cyberbullying Text Classification (NLP)

Multi-class cyberbullying detection on short social-media text using classical NLP preprocessing and TF–IDF features with a Multinomial Naive Bayes baseline. The project includes exploratory data analysis (EDA), text cleaning/normalization, model training, and evaluation (accuracy + classification report + confusion matrix).

## Dataset

This project uses the **Cyberbullying Dataset** by **Saurabh Shahane** on Kaggle:
- Dataset page: https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset

In the notebook, the dataset columns used are:
- `tweet_text` (input text)
- `cyberbullying_type` (target label)

### Expected label space

The dataset provides multiple cyberbullying categories (e.g., non-bullying vs several bullying types). See the Kaggle dataset page for the authoritative label list and licensing/usage terms.

## Project structure

```
.
├── notebooks/
│   └── Cyberbullying_NLP_1.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_nb.py
│   └── predict.py
├── data/
│   ├── raw/              # place downloaded CSV here (ignored by git)
│   └── processed/        # optional artifacts (ignored by git)
├── reports/
│   └── figures/          # optional plots (ignored by git)
├── requirements.txt
├── LICENSE
└── README.md
```

## Methods

### 1) Cleaning & preprocessing
The pipeline in `preprocess.py` mirrors the notebook:
- lowercasing
- URL / mention / hashtag normalization
- punctuation & digit removal (configurable)
- whitespace normalization
- lemmatization with `WordNetLemmatizer`
- optional stopword removal

### 2) Feature extraction
- **TF–IDF** vectorization (`sklearn.feature_extraction.text.TfidfVectorizer`)
- English stopword filtering via scikit-learn

### 3) Model(s)
- **Multinomial Naive Bayes** (`sklearn.naive_bayes.MultinomialNB`)

The notebook also imports `CountVectorizer` to enable bag-of-words experiments, but the default training script uses TF–IDF because it generally performs better for sparse text classification.

### 4) Evaluation
- accuracy
- per-class precision/recall/F1 via `classification_report`
- confusion matrix visualization

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Download the dataset
Download the Kaggle dataset and place the CSV under:
```
data/raw/cyberbullying_tweets.csv
```

If you use the Kaggle CLI:
```bash
kaggle datasets download -d saurabhshahane/cyberbullying-dataset -p data/raw --unzip
```

### 4) Train the baseline model
```bash
python -m src.train_nb --data data/raw/cyberbullying_tweets.csv --text-col tweet_text --label-col cyberbullying_type
```

Artifacts are saved under `models/` (vectorizer + model) and `outputs/` (metrics).

### 5) Run inference on new text
```bash
python -m src.predict --model models/nb_model.joblib --vectorizer models/tfidf_vectorizer.joblib --text "your text here"
```

## Reproducibility

- Train/test split: `test_size=0.2`, `random_state=42` (as in the notebook).
- If you modify preprocessing or vectorizer parameters, results will change. Keep those configs version-controlled.

## Notes on responsible use

Cyberbullying detection is a high-impact classification task. Models can exhibit bias and may produce false positives/negatives depending on dialect, reclaimed language, sarcasm, or context. Treat this repository as an educational baseline and validate carefully before any real-world use.

## Acknowledgements

- Kaggle dataset: Saurabh Shahane — Cyberbullying Dataset (see link above).
- Libraries: scikit-learn, NLTK, pandas, matplotlib, plotly.
