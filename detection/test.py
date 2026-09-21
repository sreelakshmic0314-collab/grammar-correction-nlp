"""
Evaluate the trained grammar error DETECTION model on the held-out test set.

Run: python test.py   (after running train.py)
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess

MODEL_DIR = "models"


def main():
    clf = joblib.load(os.path.join(MODEL_DIR, "grammar_classifier.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_split.csv"))

    # Same preprocessing + vectorization must be applied to test data —
    # using the vectorizer's .transform() (NOT .fit_transform()) so the
    # test set is mapped into the SAME vocabulary/weights learned from
    # training, rather than leaking test-set statistics into the model.
    test_df["clean_sentence"] = test_df["sentence"].apply(preprocess)
    X_test = vectorizer.transform(test_df["clean_sentence"])
    y_test = test_df["label"]

    y_pred = clf.predict(X_test)

    print("=" * 60)
    print("GRAMMAR ERROR DETECTION — TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.3f}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Ungrammatical", "Grammatical"]
        )
    )

    # Confusion matrix: rows = actual class, columns = predicted class.
    # Diagonal = correct predictions; off-diagonal = mistakes, split into
    # false positives and false negatives so you can discuss which kind
    # of error the model makes more often.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ungrammatical", "Grammatical"],
        yticklabels=["Ungrammatical", "Grammatical"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Grammar Error Detection")
    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(out_path)
    print(f"\nConfusion matrix image saved to: {out_path}")


if __name__ == "__main__":
    main()
