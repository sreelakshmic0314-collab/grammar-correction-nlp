"""
Text preprocessing utilities for the grammar error DETECTION pipeline.

Kept as its own module (rather than inline in train.py) so you can walk
your guide through each preprocessing step individually and show that you
understand what each one does and why.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK resources the first time this runs (then cached
# locally, so subsequent runs are instant and don't need internet).
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Step 1: Cleaning.
    Lowercase the text, strip digits/punctuation (but keep apostrophes,
    since "don't" vs "dont" matters for grammar), and collapse extra
    whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Step 2: Tokenization. Split the cleaned string into word tokens."""
    return word_tokenize(text)


def lemmatize_tokens(tokens):
    """
    Step 3: Lemmatization.
    Reduce each word to its dictionary/base form, e.g. "goes" -> "go",
    "children" -> "child". This shrinks the vocabulary so the model sees
    more examples per "concept" instead of treating every inflected form
    as a totally separate word.
    """
    return [lemmatizer.lemmatize(tok) for tok in tokens]


def preprocess(text: str, remove_stopwords: bool = False) -> str:
    """
    Full pipeline: clean -> tokenize -> (optionally remove stopwords)
    -> lemmatize -> rejoin into a string, ready to hand to the vectorizer.

    IMPORTANT DESIGN NOTE for your report/viva:
    Most NLP pipelines remove stopwords (is, was, do, does, the, a...) by
    default because they carry little *topic* meaning. But for GRAMMAR
    error detection specifically, stopwords are often exactly where the
    error lives (a missing "is", a wrong "a" vs "an", wrong "do"/"does"
    agreement). That's why remove_stopwords defaults to False here — this
    is a deliberate, explainable choice, not an oversight. It's a good
    point to bring up with your guide.
    """
    text = clean_text(text)
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    # Keep original word forms for grammar detection
    # tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    # Quick demo of each stage, useful for showing your guide step by step.
    sample = "She DON'T knows   the   answer!!"
    print("Original:    ", sample)
    print("Cleaned:     ", clean_text(sample))
    print("Tokenized:   ", tokenize(clean_text(sample)))
    print("Lemmatized:  ", lemmatize_tokens(tokenize(clean_text(sample))))
    print("Final output:", preprocess(sample))
