"""
Grammar Error Detection and Correction — Flask Web Application
================================================================

Uses a pretrained T5-based sequence-to-sequence model
(vennify/t5-base-grammar-correction) to detect and correct grammatical
errors in English text, and highlights exactly what changed.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import difflib
from html import escape

from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

MODEL_NAME = "vennify/t5-base-grammar-correction"

print("Loading grammar correction model... (this may take a minute on first run)")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")


def correct_grammar(text: str, num_beams: int = 5) -> str:
    """
    Run the input text through the T5 grammar-correction model.

    The model was fine-tuned specifically for grammar correction, so it
    expects inputs prefixed with "grammar: " (the same convention T5 uses
    for other tasks, e.g. "translate English to German: ...").
    """
    if not text.strip():
        return ""

    input_text = "grammar: " + text.strip()
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", truncation=True, max_length=256
    )

    outputs = model.generate(
        input_ids,
        max_length=256,
        num_beams=num_beams,
        early_stopping=True,
    )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected


def build_diff(original: str, corrected: str) -> str:
    """
    Build a word-level HTML diff between the original and corrected text,
    so the UI can highlight exactly which words were removed vs added.
    """
    original_words = original.split()
    corrected_words = corrected.split()

    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    pieces = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pieces.append(escape(" ".join(corrected_words[j1:j2])))
        elif tag == "replace":
            removed = escape(" ".join(original_words[i1:i2]))
            added = escape(" ".join(corrected_words[j1:j2]))
            pieces.append(f'<span class="del">{removed}</span>')
            pieces.append(f'<span class="add">{added}</span>')
        elif tag == "delete":
            removed = escape(" ".join(original_words[i1:i2]))
            pieces.append(f'<span class="del">{removed}</span>')
        elif tag == "insert":
            added = escape(" ".join(corrected_words[j1:j2]))
            pieces.append(f'<span class="add">{added}</span>')

    return " ".join(pieces)


def count_edits(original: str, corrected: str) -> int:
    """Count the number of edit operations (a rough proxy for 'errors found')."""
    matcher = difflib.SequenceMatcher(None, original.split(), corrected.split())
    return sum(1 for tag, *_ in matcher.get_opcodes() if tag != "equal")


@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    corrected_text = ""
    diff_html = ""
    error_count = 0
    submitted = False

    if request.method == "POST":
        original_text = request.form.get("text", "")
        submitted = True
        corrected_text = correct_grammar(original_text)
        diff_html = build_diff(original_text, corrected_text)
        error_count = count_edits(original_text, corrected_text)

    return render_template(
        "index.html",
        original_text=original_text,
        corrected_text=corrected_text,
        diff_html=diff_html,
        error_count=error_count,
        submitted=submitted,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
