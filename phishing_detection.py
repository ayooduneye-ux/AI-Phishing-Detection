"""
AI-Driven Phishing Detection System (TF-IDF + Logistic Regression)
Dataset: SMS Spam Collection (spam.csv)
Author: <Your Name>
University College Birmingham

How to run:
  1) pip install -r requirements.txt
  2) python src/phishing_detection.py

Outputs:
  Accuracy, Confusion Matrix, Classification Report (precision/recall/F1)
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text: str) -> str:
    """Light text cleaning suitable for TF-IDF."""
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", " URL ", t)     # normalise URLs
    t = re.sub(r"\S+@\S+", " EMAILADDR ", t)     # normalise email addresses
    t = re.sub(r"[^a-z\s]", " ", t)              # keep letters/spaces only
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    # Load dataset (Kaggle/UCI SMS Spam Collection typically uses latin-1 encoding)
    df = pd.read_csv("data/phishing_dataset.csv", encoding="latin-1")

    # Keep only useful columns: v1 = label, v2 = message text
    if "v1" not in df.columns or "v2" not in df.columns:
        raise ValueError(
            "Expected columns 'v1' (label) and 'v2' (message). " 
            "Your dataset columns are: " + str(list(df.columns))
        )

    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

    # Map labels to coursework-friendly names
    df["label"] = df["label"].map({"spam": "phishing", "ham": "legitimate"})
    df = df.dropna(subset=["label", "message"])

    # Clean text
    df["message"] = df["message"].apply(clean_text)

    # Train/test split (stratified keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # TF-IDF + Logistic Regression
    vectoriser = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectoriser.fit_transform(X_train)
    X_test_vec = vectoriser.transform(X_test)

    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n=== Confusion Matrix ===")
    # Order: phishing, legitimate
    print(confusion_matrix(y_test, y_pred, labels=["phishing", "legitimate"]))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
