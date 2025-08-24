# src/train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "tickets.csv")
MODEL_LOGREG_PATH = os.path.join("models", "ticket_classifier_logreg.pkl")
MODEL_SVM_PATH = os.path.join("models", "ticket_classifier_svm.pkl")

def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()  # normalize headers

    # Combine title + description
    df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)
    X = df["text"]
    y = df["label"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Common TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))

    # 4. Logistic Regression pipeline
    logreg = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    logreg.fit(X_train, y_train)
    print("\n=== Logistic Regression Report ===")
    print(classification_report(y_test, logreg.predict(X_test)))
    joblib.dump(logreg, MODEL_LOGREG_PATH)
    print(f"Logistic Regression model saved at {MODEL_LOGREG_PATH}")

    # 5. SVM with probability=True
    svm = Pipeline([
        ("tfidf", tfidf),
        ("clf", SVC(kernel="linear", probability=True))
    ])
    svm.fit(X_train, y_train)
    print("\n=== SVM Report ===")
    print(classification_report(y_test, svm.predict(X_test)))
    joblib.dump(svm, MODEL_SVM_PATH)
    print(f"SVM model saved at {MODEL_SVM_PATH}")

if __name__ == "__main__":
    main()

