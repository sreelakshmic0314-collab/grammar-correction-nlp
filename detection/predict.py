"""
Interactive CLI to try the trained detection model on your own sentences.
Useful for a live demo in front of your guide.

Run: python predict.py    (after running train.py)
"""

import os
import joblib
import math

from preprocess import preprocess

MODEL_DIR = "models"


def load_model():
    clf = joblib.load(os.path.join(MODEL_DIR, "grammar_classifier.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    return clf, vectorizer


def predict(sentence: str, clf, vectorizer) -> str:
    clean = preprocess(sentence)
    vec = vectorizer.transform([clean])
    pred = clf.predict(vec)[0]
    score = clf.decision_function(vec)[0]
    confidence = 1 / (1 + math.exp(-abs(score)))
    label = "Grammatically correct" if pred == 1 else "Grammar error detected"
    return f"{label}  (confidence: {confidence:.2f})"


if __name__ == "__main__":
    clf, vectorizer = load_model()
    print("Grammar error detector — type a sentence to check (or 'quit' to exit)\n")
    while True:
        sentence = input("> ")
        if sentence.strip().lower() in ("quit", "exit"):
            break
        if not sentence.strip():
            continue
        print(predict(sentence, clf, vectorizer))
