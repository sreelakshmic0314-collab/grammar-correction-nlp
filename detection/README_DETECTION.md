# Grammar Error Detection — Preprocessing, Vectorization, Training & Testing

This module is the classical-ML complement to the T5-based correction app
in the parent folder. Where the correction app *uses* a pretrained
Transformer, this module builds a model **from scratch** through every
stage your guide asked about: preprocessing, vectorization, training,
and testing.

**Task framing:** Grammar error *detection* is treated as **binary
sentence classification** — given a sentence, predict `1` (grammatically
acceptable) or `0` (contains a grammar error). This is a different,
simpler framing than *correction* (which generates a rewritten sentence),
and it's the standard framing used by the CoLA benchmark.

---

## 1. The dataset: CoLA

**CoLA** (Corpus of Linguistic Acceptability) is a dataset of ~10,000
English sentences, each labeled by trained linguists as grammatically
*acceptable* or *unacceptable*. It's part of the **GLUE benchmark**, a
standard suite used to evaluate NLP models. It's a good fit here because,
unlike JFLEG (which pairs an error with a rewritten correction), CoLA is
built specifically for **classification** — exactly what we need for
detection.

`train.py` downloads it automatically via Hugging Face's `datasets`
library the first time you run it (needs internet once). If you're
offline, it automatically falls back to `data/sample_labeled_sentences.csv`,
a small 50-sentence set bundled in this repo so the pipeline still runs
end-to-end for a demo. **For your actual report results, run it with
internet at least once** so you train on the real ~10,000-sentence CoLA
set — the fallback CSV is only for offline demos, not for your reported
numbers.

---

## 2. Preprocessing (`preprocess.py`)

Four explicit steps, each a separate function so you can explain them
individually:

1. **Cleaning** (`clean_text`) — lowercase the text, strip digits/most
   punctuation, collapse repeated whitespace. Apostrophes are kept
   ("don't" vs "dont" is grammatically meaningful).
2. **Tokenization** (`tokenize`) — split the cleaned string into
   individual word tokens using NLTK's tokenizer.
3. **Stopword removal** — *optional* and **off by default**. Explain to
   your guide why: stopwords like "is", "was", "do", "does" are usually
   removed in topic-classification tasks, but for *grammar* detection
   they're often the error itself (e.g. a missing "is"). Removing them
   would throw away the signal we're trying to detect.
4. **Lemmatization** (`lemmatize_tokens`) — reduce words to their base
   dictionary form (e.g. "goes" → "go"), shrinking the vocabulary so the
   model sees more examples per underlying word.

Run `python preprocess.py` standalone to see each stage printed for a
sample sentence — good for a live walkthrough.

---

## 3. Vectorization (`train.py`, step 2)

Machine learning models need numbers, not text, so each preprocessed
sentence is converted into a numeric vector using **TF-IDF**
(Term Frequency – Inverse Document Frequency):

- **Term Frequency:** how often a word/n-gram appears *in this sentence*.
- **Inverse Document Frequency:** a down-weighting for words that appear
  in *many* sentences across the dataset (like "the"), since they carry
  little distinguishing signal.
- The product of the two gives each word a weight that's high when a
  word is frequent *here* but rare *overall* — i.e. distinctive.

We use `ngram_range=(1, 2)`, meaning the vectorizer captures both single
words (unigrams) **and** word pairs (bigrams). This matters for grammar
detection specifically, because many errors are only visible in a pair of
adjacent words (e.g. "he go", "a apple", "many reason") — a single word
in isolation ("go") isn't wrong, only its pairing with "he" is.

This is a good point to contrast with the correction app: TF-IDF vectors
are **sparse, hand-designed, frequency-based features**, versus the T5
model's **dense, learned embeddings** from Transformer pretraining. That
contrast is worth a paragraph in your report.

---

## 4. Training (`train.py`, step 3)

We train a **Logistic Regression** classifier on the TF-IDF vectors.
Why Logistic Regression:
- Fast to train, no GPU needed.
- Highly interpretable: each TF-IDF feature (word/bigram) gets one
  learned weight; positive weight pushes toward "grammatical", negative
  pushes toward "error". You can literally inspect and show the
  top weighted words if your guide asks how the model "knows".
- A standard, respected baseline in NLP coursework before moving to
  neural methods.

`class_weight="balanced"` adjusts for any imbalance between the number of
"acceptable" vs "unacceptable" examples. The trained model and the fitted
vectorizer are saved to `models/` with `joblib`, along with the held-out
test split, so training and testing are cleanly separated.

---

## 5. Testing / Evaluation (`test.py`)

The held-out test set (sentences the model never saw during training) is
run through the **same** preprocessing and the **same fitted** vectorizer
(using `.transform()`, not `.fit_transform()`, so no test-set information
leaks into the vocabulary/weights). We report:

- **Accuracy** — overall fraction correct.
- **Precision** — of sentences predicted "grammatical", how many actually
  were.
- **Recall** — of sentences that actually were "grammatical", how many
  the model caught.
- **F1 score** — harmonic mean of precision and recall, useful when
  classes are imbalanced.
- **Confusion matrix** — saved as `models/confusion_matrix.png`, shows
  the split between correct predictions, false positives, and false
  negatives.

---

## 6. How to run it

```bash
cd detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py      # preprocess + vectorize + train, saves model to models/
python test.py        # evaluate on held-out test set, prints metrics + confusion matrix
python predict.py     # try your own sentences interactively
```

---

## 7. How this connects to the correction app

Together, the two modules cover the full "detection and correction"
brief:

| | This module (`detection/`) | Parent app (`app.py`) |
|---|---|---|
| Task | Binary classification: error or not | Generation: rewrite the sentence |
| Dataset | CoLA | JFLEG (used to pretrain the model we load) |
| Features | TF-IDF (hand-designed, sparse) | Learned Transformer embeddings (dense) |
| Model | Logistic Regression (trained by you, from scratch) | Pretrained T5 (used via transfer learning) |
| What it teaches | Classical NLP pipeline: preprocessing → vectorization → training → testing | Transfer learning / using pretrained models |

A natural extension (mention this as "future work" in your report): chain
them — run the detector first, and only call the T5 correction model when
the detector flags a sentence as containing an error, rather than always
correcting every input.

---

## 8. Talking points for your guide

- Why detection is framed as classification, while correction is framed
  as generation — and why that means they need different datasets.
- Why TF-IDF with bigrams, not just unigrams.
- Why stopwords are *kept* here, against the usual default.
- Why Logistic Regression is interpretable, and what "the model learns a
  weight per feature" actually means.
- Precision vs recall trade-off: would you rather the tool miss an error
  (false negative) or flag a correct sentence as wrong (false positive)?
  Argue a side — this is a common viva question.
- What you'd try next with more time: SVM or Naive Bayes as a comparison
  model, POS-tag-based features, or a small neural classifier (e.g. an
  LSTM) trained on the same data.
