# src/utils/store.py
"""Small JSON store for the state that must outlive a restart.

Alarms, timers and reminders were `threading.Timer` objects and nothing else, so
restarting JANET silently threw them away. That is the same class of problem as
a fabricated action claim: JANET told you the alarm was set, and it *was* — right
up until it wasn't, and you found out by oversleeping. Anything JANET promises
for a future time has to survive the process that promised it.

Deliberately a JSON file per feature, not a database:
- you can read it, and delete a stuck entry with a text editor,
- there is no schema migration to get wrong,
- and it fails quietly. A store that crashes JANET on a malformed file would be
  worse than one that loses an alarm, so every error here is swallowed with a
  warning (the same posture as `ai_core/transcript.py`).

Writes are atomic (temp file + `os.replace`) because the background `Timer`
threads write from outside the main thread, and a half-written file read at the
next boot would lose everything rather than one entry.
"""
import json
import os
import tempfile
import threading
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[2] / "data" / "state"

# One lock for all files: writes are tiny and rare (only when you set or cancel
# something), so a single lock is simpler than one per name and costs nothing.
_lock = threading.Lock()


def path_for(name):
    return STATE_DIR / f"{name}.json"


def load(name, default=None):
    """Read `name`.json, or return `default` if it's missing or unreadable."""
    try:
        with open(path_for(name), encoding="utf-8") as handle:
            value = json.load(handle)
        # Valid JSON of the WRONG SHAPE is its own failure: a file holding a bare
        # string parsed fine, and the caller then iterated its characters and
        # logged ten warnings. Shape is part of being readable.
        if default is not None and not isinstance(value, type(default)):
            print(f"⚠  saved {name} has the wrong shape — starting empty")
            return default
        return value
    except FileNotFoundError:
        return default
    except RecursionError:                  # pathological nesting
        print(f"⚠  saved {name} is malformed — starting empty")
        return default
    except (OSError, ValueError) as exc:
        # Corrupt or unreadable: start clean rather than refusing to boot.
        print(f"⚠  couldn't read saved {name} ({exc}) — starting empty")
        return default


def save(name, value):
    """Write `value` to `name`.json atomically. Never raises."""
    try:
        with _lock:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            # Write beside the target so os.replace stays on one filesystem,
            # which is what makes it atomic.
            fd, tmp = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, indent=2)
                os.replace(tmp, path_for(name))
            except BaseException:
                # Don't leave a temp file behind on any failure path.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except OSError as exc:
        print(f"⚠  couldn't save {name}: {exc}")
