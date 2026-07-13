"""Load the JANET intent dataset into tokenized tensors for DistilBERT.

One concern: turn `data/text/{split}/*.txt` into a torch Dataset of
(input_ids, attention_mask, label). It mirrors the format contract that
`data/text/validate.py` enforces:
  - the label is the FILENAME (TIME.txt -> label "TIME"),
  - blank lines and lines starting with "#" are ignored,
  - every remaining line is one example utterance.

`labels.txt` is the source of truth for the label set AND their order. A
label's position in that file is its integer class id for the model, so the
mapping stays stable as long as labels.txt is unchanged.
"""
from pathlib import Path

import torch

# Resolve the data dir from THIS file's location (parents[2] == repo root),
# so the loader works no matter which directory python is launched from.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "text"


def load_labels():
    """Return the 13 labels in file order. Index in this list == class id."""
    lines = (DATA_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def read_split(split, label2id):
    """Return (texts, label_ids) for split == "train" or "val".

    Reads one file per label and skips comments/blanks, exactly like the
    validator does, so training sees precisely the utterances we validated.
    """
    texts, labels = [], []
    for label, idx in label2id.items():
        path = DATA_DIR / split / f"{label}.txt"
        for raw in path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            texts.append(s)
            labels.append(idx)
    return texts, labels


class IntentDataset(torch.utils.data.Dataset):
    """Holds pre-tokenized encodings + labels; yields one dict per example.

    The HuggingFace Trainer expects each item to be a dict of tensors with a
    "labels" key. We tokenize the whole split once up front (cheap for ~2k
    short lines) rather than per __getitem__ call.
    """

    def __init__(self, texts, labels, tokenizer, max_length=32):
        # Utterances are short spoken commands, so 32 tokens is plenty of room
        # and keeps each batch small/fast. Pad every line to the same length.
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item
