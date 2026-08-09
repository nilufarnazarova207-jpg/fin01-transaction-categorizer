"""
FIN-01 — Automatic Transaction Categorization
Reproducible training pipeline.

Downloads the Wells Fargo Campus Analytics Challenge dataset (40,000 labeled
bank transactions), cleans the text, trains a TF-IDF + Logistic Regression
classifier, evaluates it, and exports the trained model for both Python reuse
and browser-based (JavaScript) inference.

Usage:
    pip install -r requirements.txt
    python train.py
"""

import os
import re
import json
import time
import urllib.request

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

DATA_URL = "https://raw.githubusercontent.com/utribedi/Bank_transaction_category_predictor/main/Training.xlsx"
DATA_PATH = "data/Training.xlsx"
MODEL_DIR = "model"
CONFIDENCE_THRESHOLD = 0.4

# Categories merged into a single "Other Services" bucket because each had
# fewer than 300 training examples and, individually, produced very poor
# precision (~0.13-0.17) despite reasonable recall under class-balanced
# weighting. Merging them turns three unreliable rare classes into one
# usable one. See docs/limitations_and_risks.md for the full discussion.
RARE_CLASS_MERGE = {
    "Communication Services": "Other Services",
    "Finance": "Other Services",
    "Education": "Other Services",
}


def clean_text(s: str) -> str:
    """Strip boilerplate/noise tokens from a raw transaction description."""
    s = str(s).upper()
    s = re.sub(r"CHECK\s*CRD\s*PURCHASE", " ", s)
    s = re.sub(r"HSA\s*CARD\s*PURCHASE", " ", s)
    s = re.sub(r"RECURRING\s*PMT", " ", s)
    s = re.sub(r"\bMCC\b", " ", s)
    s = re.sub(r"[^A-Z\s]", " ", s)          # strip digits/punctuation
    s = re.sub(r"\bX{3,}\b", " ", s)          # masked card-number remnants
    s = re.sub(r"\s+", " ", s).strip()
    return s


def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    else:
        print("Dataset already present, skipping download.")


def main():
    download_data()

    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} transactions.")

    # Combine transaction description with the coalesced brand name
    # (brand field has no missing values and adds useful merchant signal
    # that the raw description sometimes lacks or abbreviates).
    df["text"] = df["trans_desc"].astype(str) + " " + df["coalesced_brand"].astype(str)
    df["Category2"] = df["Category"].replace(RARE_CLASS_MERGE)
    df["clean_text"] = df["text"].apply(clean_text)

    # Stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["Category2"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Category2"], random_state=42
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    vectorizer = TfidfVectorizer(
        max_features=6000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train, y_val, y_test = train_df["Category2"], val_df["Category2"], test_df["Category2"]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=5.0)
    start = time.time()
    clf.fit(X_train, y_train)
    print(f"Trained in {time.time() - start:.1f}s")

    val_preds = clf.predict(X_val)
    print("\n=== VALIDATION ===")
    print("Macro F1:", f1_score(y_val, val_preds, average="macro"))
    print("Weighted F1:", f1_score(y_val, val_preds, average="weighted"))
    print(classification_report(y_val, val_preds, zero_division=0))

    test_preds = clf.predict(X_test)
    print("\n=== FINAL TEST (held out) ===")
    print("Macro F1:", f1_score(y_test, test_preds, average="macro"))
    print("Weighted F1:", f1_score(y_test, test_preds, average="weighted"))
    print(classification_report(y_test, test_preds, zero_division=0))

    # Confidence-threshold analysis (coverage vs. accuracy tradeoff)
    probs = clf.predict_proba(X_test)
    max_conf = probs.max(axis=1)
    preds = clf.classes_[probs.argmax(axis=1)]
    print("=== CONFIDENCE THRESHOLD ANALYSIS ===")
    for t in [0.0, 0.3, 0.4, 0.5, 0.6]:
        accepted = max_conf >= t
        coverage = accepted.mean()
        acc = accuracy_score(np.array(y_test)[accepted], preds[accepted]) if accepted.sum() else float("nan")
        print(f"threshold={t:.1f} | coverage={coverage:.1%} | accuracy on accepted={acc:.3f}")

    # ===== Save artifacts =====
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer_v2.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf_v2.joblib"))

    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    export = {
        "vocab": vocab,
        "idf": [float(x) for x in vectorizer.idf_],
        "classes": [str(c) for c in clf.classes_],
        "coef": [[float(x) for x in row] for row in clf.coef_],
        "intercept": [float(x) for x in clf.intercept_],
        "ngram_range": [1, 2],
        "threshold": CONFIDENCE_THRESHOLD,
    }
    with open(os.path.join(MODEL_DIR, "model_export.json"), "w") as f:
        json.dump(export, f)

    print(f"\nSaved model artifacts to {MODEL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
