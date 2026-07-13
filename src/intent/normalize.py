"""Canonicalize an STT transcript to the training format.

Whisper emits capitals, punctuation, and contractions ("What's the time?"), but
our dataset — and therefore the scorer's word matching and the classifier's
tokenizer — assume the clean form the format contract enforces: lowercase, no
punctuation, apostrophes removed ("whats the time"). Normalizing once at the
pipeline entry fixes the scorer's matching AND removes the classifier's
train/serve skew (it was trained on "whats", fed "What's").
"""
import re

_APOSTROPHE = re.compile(r"[’']")       # what's -> whats, don't -> dont
_NON_WORD = re.compile(r"[^\w\s]")      # other punctuation ( . , ? ! - " ) -> space; keeps digits


def normalize(text):
    """Return text lowercased, apostrophe-free, punctuation-stripped, single-spaced."""
    text = text.lower()
    text = _APOSTROPHE.sub("", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())
