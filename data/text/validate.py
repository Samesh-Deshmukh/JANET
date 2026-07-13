#!/usr/bin/env python3
"""Validate the JANET intent dataset against its format contract.

Run: python data/text/validate.py   (exit 0 = ok, 1 = violations)
Standalone: stdlib only, no src imports, resolves paths from its own location.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val")
FORBIDDEN = set('.,?!;:"()[]{}')


def load_labels():
    return [ln.strip() for ln in (HERE / "labels.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]


def utterances(path):
    """Non-blank, non-comment lines from a data file (raw, not lowered)."""
    if not path.exists():
        return None
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main():
    labels = load_labels()
    problems = []
    counts = {}
    for label in labels:
        per_split = {}
        for split in SPLITS:
            lines = utterances(HERE / split / f"{label}.txt")
            if lines is None:
                problems.append(f"MISSING FILE: {split}/{label}.txt")
                per_split[split] = []
                continue
            for ln in lines:
                if ln != ln.lower():
                    problems.append(f"{split}/{label}.txt: not lowercase: {ln!r}")
                bad = FORBIDDEN & set(ln)
                if bad:
                    problems.append(f"{split}/{label}.txt: forbidden char(s) {sorted(bad)}: {ln!r}")
            dupes = {x for x in lines if lines.count(x) > 1}
            if dupes:
                problems.append(f"{split}/{label}.txt: duplicate line(s): {sorted(dupes)[:3]}")
            per_split[split] = lines
        overlap = set(per_split["train"]) & set(per_split["val"])
        if overlap:
            problems.append(f"{label}: {len(overlap)} line(s) in BOTH train and val, e.g. {sorted(overlap)[:3]}")
        counts[label] = (len(per_split["train"]), len(per_split["val"]))

    print(f"{'label':<12}{'train':>8}{'val':>8}{'total':>8}")
    t_tr = t_va = 0
    for label in labels:
        tr, va = counts[label]
        t_tr += tr
        t_va += va
        print(f"{label:<12}{tr:>8}{va:>8}{tr+va:>8}")
    print(f"{'TOTAL':<12}{t_tr:>8}{t_va:>8}{t_tr+t_va:>8}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):", file=sys.stderr)
        for p in problems[:50]:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    print("\nOK: format contract satisfied, train/val disjoint.")


if __name__ == "__main__":
    main()
