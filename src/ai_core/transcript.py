# src/ai_core/transcript.py
"""Append-only record of everything JANET decided, one JSON object per turn.

Holds what the in-memory history can't: the facts an action returned, the
reasoning behind the reply, and any deep-thinking trace — kept on disk so a
conversation can be reviewed (or the scorer/classifier tuned) long after the
process has exited.

Writing must never break the assistant, so every I/O error is swallowed with a
warning. A missing log line is annoying; a crash mid-conversation is not.
"""
import json
from datetime import datetime
from pathlib import Path

# src/ai_core/transcript.py -> src/ai_core -> src -> the repo root
TRANSCRIPT_DIR = Path(__file__).resolve().parents[2] / "data" / "transcripts"


def path_for(when=None):
    """One file per day, so a day's conversation is easy to read back."""
    when = when or datetime.now()
    return TRANSCRIPT_DIR / f"{when:%Y-%m-%d}.jsonl"


def log(**entry):
    """Append one turn. Returns True if it was written, False if it wasn't."""
    try:
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **entry}
        with path_for().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        print(f"⚠  transcript write failed: {exc}")
        return False
