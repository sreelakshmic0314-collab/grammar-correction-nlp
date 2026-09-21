"""
Evaluation script for the grammar correction model.
=====================================================

Computes the GLEU score — the standard metric used in the JFLEG benchmark
for Grammatical Error Correction — comparing the model's corrections
against reference (human-corrected) sentences on a small hand-picked
evaluation set.

Use these numbers in your project report's "Results" section.

Run: python evaluate.py
"""

from nltk.translate.gleu_score import sentence_gleu

from app import correct_grammar

# A small evaluation set in the style of JFLEG / CoNLL-2014.
# Each entry is (sentence with errors, [one or more acceptable corrections]).
TEST_SET = [
    ("She go to school everyday.", ["She goes to school every day."]),
    ("He dont know the answer.", ["He doesn't know the answer."]),
    ("I have went to the market yesterday.", ["I went to the market yesterday."]),
    ("There is many reason for this.", ["There are many reasons for this."]),
    ("Me and my friend was late for class.", ["My friend and I were late for class."]),
    ("She is more taller than her brother.", ["She is taller than her brother."]),
    ("I look forward to hear from you.", ["I look forward to hearing from you."]),
    ("The information is very usefull for us.", ["The information is very useful for us."]),
    ("Each of the students have their own laptop.", ["Each of the students has their own laptop."]),
    ("He said that he is coming tomorrow.", ["He said that he was coming tomorrow."]),
]


def evaluate() -> None:
    scores = []

    print(f"{'Original':45} | {'Model output':45} | GLEU")
    print("-" * 105)

    for source, references in TEST_SET:
        prediction = correct_grammar(source)

        tokenized_refs = [ref.split() for ref in references]
        tokenized_pred = prediction.split()

        score = sentence_gleu(tokenized_refs, tokenized_pred)
        scores.append(score)

        print(f"{source:45} | {prediction:45} | {score:.3f}")

    avg_score = sum(scores) / len(scores)
    print("-" * 105)
    print(f"Average GLEU score over {len(TEST_SET)} sentences: {avg_score:.3f}")
    print(
        "\nNote: GLEU ranges from 0 to 1. Published GEC systems on JFLEG "
        "typically score in the 0.4-0.6 range, so use this as a rough "
        "sanity check rather than a strict benchmark comparison, since "
        "this evaluation set is tiny and hand-picked."
    )


if __name__ == "__main__":
    evaluate()
