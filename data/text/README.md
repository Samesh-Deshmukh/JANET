# JANET intent dataset

Training data for the Block 3 intent classifier. See the design at
`docs/superpowers/specs/2026-07-13-intent-training-dataset-design.md`.

## Layout
- `labels.txt` — the 13 labels, source of truth.
- `train/LABEL.txt`, `val/LABEL.txt` — one file per label per split.

## Format contract
- One utterance per line. **The label is the filename** — no inline labels.
- Natural **lowercase**, **no sentence punctuation** (`. , ? ! ; : " ( ) [ ] { }`
  are forbidden). Apostrophes, hyphens, digits, spaces are allowed. This mirrors
  real Whisper `tiny` output so training matches inference.
- Blank lines and lines starting with `#` are ignored (used for grouping).
- `train/` and `val/` for the same label must be disjoint (no shared line).

## Validate
`python data/text/validate.py` — checks the contract and prints per-label counts.
Exit 0 = clean.
