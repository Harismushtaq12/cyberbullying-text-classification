"""Train a Multinomial Naive Bayes baseline with TF–IDF features.

Example:
    python -m src.train_nb --data data/raw/cyberbullying_tweets.csv \
        --text-col tweet_text --label-col cyberbullying_type
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .preprocess import batch_clean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV dataset file")
    p.add_argument("--text-col", default="tweet_text", help="Name of text column")
    p.add_argument("--label-col", default="cyberbullying_type", help="Name of label column")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-features", type=int, default=None, help="Optional TF–IDF max_features")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--outputs-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{args.text_col}' and '{args.label_col}'. "
            f"Found: {list(df.columns)}"
        )

    texts = batch_clean(df[args.text_col].astype(str).tolist())
    labels = df[args.label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    labels_sorted = sorted(set(labels))

    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "nb_model.joblib")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels_sorted,
        "config": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "vectorizer": {"stop_words": "english", "max_features": args.max_features},
            "model": "MultinomialNB",
        },
    }
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy: {acc:.4f}")
    print("Saved:")
    print(f"  model -> {models_dir / 'nb_model.joblib'}")
    print(f"  vectorizer -> {models_dir / 'tfidf_vectorizer.joblib'}")
    print(f"  metrics -> {outputs_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
