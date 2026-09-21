"""
Train a grammar error DETECTION model (binary classifier).

Pipeline:  load data -> preprocess -> vectorize (TF-IDF) -> train
           classifier -> save model + vectorizer to disk for testing.

Run: python train.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from preprocess import preprocess

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """
    Load the CoLA (Corpus of Linguistic Acceptability) dataset.

    CoLA contains ~10,000 English sentences, each hand-labeled by trained
    linguists as grammatically ACCEPTABLE (1) or UNACCEPTABLE (0). It's
    part of the GLUE benchmark and is *the* standard academic dataset for
    this exact task: sentence-level grammatical acceptability.

    Requires internet on first run (downloads once, then caches locally).
    Falls back to a small bundled CSV if offline, so the pipeline still
    runs end-to-end for a demo.
    """
    try:
        from datasets import load_dataset
        print("Downloading CoLA dataset from Hugging Face (glue/cola)...")
        dataset = load_dataset("nyu-mll/glue", "cola")
        train_df = dataset["train"].to_pandas()[["sentence", "label"]]
        test_df = dataset["validation"].to_pandas()[["sentence", "label"]]
        print(f"Loaded CoLA: {len(train_df)} train / {len(test_df)} validation sentences.")
        return train_df, test_df
    except Exception as e:
        print(f"Could not download CoLA ({e}).")
        print("Falling back to the small bundled sample dataset in data/.")
        df = pd.read_csv("data/sample_labeled_sentences.csv")
        train_df, test_df = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df["label"]
        )
        return train_df, test_df


def main():
    train_df, test_df = load_data()

    # ---------------- Preprocessing ----------------
    print("\n[1/3] Preprocessing sentences...")
    train_df["clean_sentence"] = train_df["sentence"].apply(preprocess)
    test_df["clean_sentence"] = test_df["sentence"].apply(preprocess)

    # ---------------- Vectorization ----------------
    # TF-IDF (Term Frequency - Inverse Document Frequency) turns each
    # sentence into a numeric vector: one dimension per word/n-gram in the
    # vocabulary. A word gets a HIGH weight in a sentence if it appears
    # often IN that sentence but RARELY across the whole dataset — i.e. it
    # is distinctive. Common words like "the" that appear everywhere end
    # up down-weighted automatically, without needing a stopword list.
    #
    # ngram_range=(1,2) keeps single words AND word-pairs (bigrams),
    # because grammar errors are often only visible in a pair of adjacent
    # words (e.g. "he go", "many reason", "a apple") — a bag of single
    # words alone would miss that "go" next to "he" is the problem.
    print("[2/3] Vectorizing with TF-IDF (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train = vectorizer.fit_transform(train_df["clean_sentence"])
    y_train = train_df["label"]

    # ---------------- Training ----------------
    # Logistic Regression: a simple, fast, highly interpretable linear
    # classifier — a good baseline for text classification and easy to
    # explain in a viva (it learns a weight per TF-IDF feature; positive
    # weight pushes toward "grammatical", negative pushes toward "error").
    # class_weight="balanced" compensates if one class has more examples.
    print("[3/3] Training Logistic Regression classifier...")
    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    print(f"\nTraining accuracy: {train_accuracy:.3f}")

    # ---------------- Save artifacts ----------------
    joblib.dump(clf, os.path.join(MODEL_DIR, "grammar_classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    test_df.to_csv(os.path.join(MODEL_DIR, "test_split.csv"), index=False)

    print("\nSaved to models/:")
    print("  - grammar_classifier.joblib   (trained model)")
    print("  - tfidf_vectorizer.joblib     (fitted vectorizer)")
    print("  - test_split.csv              (held-out test data)")
    print("\nNext: run 'python test.py' to evaluate on the held-out test set.")


if __name__ == "__main__":
    main()
