"""Run inference using a saved TF–IDF vectorizer + MultinomialNB model."""

from __future__ import annotations

import argparse

import joblib

from .preprocess import clean_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved model joblib")
    p.add_argument("--vectorizer", required=True, help="Path to saved vectorizer joblib")
    p.add_argument("--text", required=True, help="Input text to classify")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = joblib.load(args.model)
    vectorizer = joblib.load(args.vectorizer)

    t = clean_text(args.text)
    X = vectorizer.transform([t])
    pred = model.predict(X)[0]

    # Some sklearn models support predict_proba; Naive Bayes does.
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].max()

    if proba is None:
        print(f"Prediction: {pred}")
    else:
        print(f"Prediction: {pred} (confidence={proba:.3f})")


if __name__ == "__main__":
    main()
