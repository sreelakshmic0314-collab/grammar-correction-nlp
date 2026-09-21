# Grammar Error Detection and Correction

An NLP web app that detects and corrects grammatical errors in English text
using a pretrained T5 (Text-to-Text Transfer Transformer) model, with a
word-level diff view that highlights exactly what was changed.

---

## 1. How it works (for your report / viva)

**Task framing.** Grammatical Error Correction (GEC) is framed here as a
**sequence-to-sequence** problem: the model reads a sentence that may
contain errors and generates a corrected sentence, the same way a
translation model reads a sentence in one language and generates its
translation in another.

**Model.** We use [`vennify/t5-base-grammar-correction`](https://huggingface.co/vennify/t5-base-grammar-correction),
a T5-base model that has already been fine-tuned specifically for grammar
correction (T5 = an encoder-decoder Transformer originally from Google's
"Exploring the Limits of Transfer Learning" paper). Because it's already
fine-tuned, you get strong results without training anything yourself —
that's the trade-off you chose (pretrained model vs. training your own).

**Inference pipeline** (see `app.py`):
1. The input sentence is prefixed with the task instruction `"grammar: "`,
   matching how the model was trained (T5 uses text prefixes to indicate
   the task, e.g. `"translate English to German: "`).
2. The tokenizer converts the text into token IDs (`T5Tokenizer`, a
   SentencePiece-based subword tokenizer).
3. `model.generate()` performs **beam search** (`num_beams=5`) to decode
   the most likely corrected sentence.
4. The output token IDs are decoded back into text.

**Diff highlighting.** `difflib.SequenceMatcher` (Python's standard-library
diffing algorithm, the same family of algorithm used by `diff`/`git diff`)
compares the original and corrected sentences word-by-word and classifies
each span as `equal`, `replace`, `insert`, or `delete`. The web page then
renders deletions with strikethrough and insertions highlighted, so a
grader can see at a glance what the model actually fixed.

**Evaluation.** `evaluate.py` computes **GLEU** (Google-BLEU), the standard
automatic metric for GEC used by the JFLEG benchmark, which rewards
matching the reference correction while, unlike BLEU, also penalizing
words that were "over-corrected" beyond the reference. Use this script's
output in your report's results section.

---

## 2. Project structure

```
grammar-correction-project/
├── app.py                      # Flask app: model loading, correction, diffing, routes
├── evaluate.py                 # GLEU-score evaluation script for your report
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PROJECT_REPORT_OUTLINE.md   # Suggested structure for your academic report
├── templates/
│   └── index.html              # Web page (Jinja2 template)
├── static/
│   └── style.css                # Styling
└── sample_data/
    └── test_sentences.txt      # Example sentences to paste into the app for a demo
```

---

## 3. Setup instructions

**Requirements:** Python 3.9–3.11, ~2 GB free disk space (for the model
weights, downloaded once and cached), and an internet connection for the
first run only.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The first run will download the model (~850 MB) from Hugging Face Hub and
cache it locally (`~/.cache/huggingface`), so it only happens once.

---

## 4. Using the app

1. Type or paste a sentence with grammar mistakes into the text box (or
   copy one from `sample_data/test_sentences.txt`).
2. Click **Check grammar**.
3. The page shows:
   - How many changes were made.
   - A diff view: struck-through red = removed, highlighted green = added.
   - The final corrected sentence.

---

## 5. Running the evaluation

```bash
python evaluate.py
```

This prints a table comparing the model's output to reference corrections
for 10 hand-picked sentences, along with the GLEU score for each and the
average — a quick, quantitative result you can quote directly in your
report.

---

## 6. Known limitations (worth mentioning in your report)

- **Sentence-level, not document-level:** works best on individual
  sentences; long paragraphs may need to be split first.
- **No error-type labels:** the model rewrites the sentence but doesn't
  classify errors by type (subject-verb agreement, tense, article usage,
  etc.). This is a natural "future work" extension — see below.
- **Beam search is deterministic but not always minimal:** the model
  sometimes rewrites more of the sentence than strictly necessary to fix
  the error, which is why the diff view is useful for transparency.
- **English only,** and works best on the kind of learner-English errors
  it was fine-tuned on (article/preposition/tense/agreement errors).

## 7. Ideas for extending this project

- Add **error-type classification** (e.g. using POS tagging with spaCy to
  label *why* a change was made — subject-verb agreement, tense, etc.).
- **Fine-tune your own model** on a GEC dataset like JFLEG, Lang-8, or the
  C4_200M synthetic GEC corpus, and compare its GLEU score against this
  pretrained baseline — a strong way to turn this into a more advanced
  project or extend it for a thesis.
- Add a **rule-based comparison** using `language_tool_python` (a Python
  wrapper for LanguageTool) alongside the neural model, and discuss the
  trade-offs between rule-based and neural approaches in your report.
- Support **multi-sentence / paragraph input** by splitting on sentence
  boundaries (e.g. with `nltk.sent_tokenize`) before correcting each one.

---

## 8. Suggested viva/demo talking points

- Why sequence-to-sequence (rather than classification) is the natural
  framing for grammar correction.
- What a pretrained model gives you vs. training from scratch (transfer
  learning, why it needs much less data/compute).
- What beam search is and why `num_beams=5` is used instead of greedy
  decoding.
- Why GLEU rather than plain BLEU is used for GEC evaluation.
- The limitations above, and how you'd address them with more time.
