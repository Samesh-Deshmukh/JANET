"""Load the fine-tuned intent model once and predict a label for an utterance.

One concern: given a transcript string, return (label, confidence). The model
and tokenizer are cached in a module-level singleton (same pattern as stt.py's
Whisper cache) so the ~268MB DistilBERT loads a single time, not per call.

The model is produced by `python -m intent.train`; its config already carries
the id -> label names, so we don't re-read labels.txt here.
"""
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "intent-distilbert"

_cache = {}  # lazily filled with {"tok", "model", "device"}


def _load():
    """Load (once) and return the cached tokenizer/model/device."""
    if not _cache:
        if not _MODEL_DIR.exists():
            raise FileNotFoundError(
                f"intent model not found at {_MODEL_DIR}. "
                "Train it first: cd src && python -m intent.train"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["tok"] = AutoTokenizer.from_pretrained(_MODEL_DIR)
        _cache["model"] = (
            AutoModelForSequenceClassification.from_pretrained(_MODEL_DIR).to(device).eval()
        )
        _cache["device"] = device
    return _cache


def predict(text):
    """Return (label, confidence) for one utterance.

    confidence is the softmax probability of the winning class in [0, 1].
    """
    c = _load()
    inputs = c["tok"](text, return_tensors="pt", truncation=True, max_length=32).to(c["device"])
    with torch.no_grad():
        logits = c["model"](**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return c["model"].config.id2label[idx], float(probs[idx])
