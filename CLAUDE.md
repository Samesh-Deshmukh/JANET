# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JANET (Just Another Neural Execution Tool) is a fully-local, privacy-first voice assistant. The repo is at an **early prototype stage** that implements a tiny slice of a much larger design.

- `Notes/JANET_v2.0.md` is the authoritative ~2400-line system design spec (audio pipeline, vision, intent router, 3-layer AI core, context manager, integrations, etc.). Treat it as the north-star architecture. **Note:** `Notes/` is gitignored, so it exists only in the local working copy — don't assume other clones have it.
- `src/` currently implements only push-to-talk capture → transcription → keyword intent → speech. The packages `src/ai_core/`, `src/vision/`, and `src/integrations/` are empty `__init__.py` placeholders reserved for design-doc components not yet built.

When adding features, check the corresponding section of the design doc first — most "new" work is already specified there.

## Running

Requires audio hardware (mic + speaker) and, on Linux, permission for the `keyboard` library to read global key state (typically root or input-group access).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src && python main.py     # must run from src/ — see import note below
```

At runtime: hold **space** to record; release to stop. Audio is written to `output.wav` in the current directory, transcribed, and the intent is printed. Ctrl-C to exit the loop.

- Python 3.11 (the committed `venv/` targets 3.11).
- There is no build step, linter, or test suite configured. `tests/` is gitignored scratch space (`t.py`, `temp.py`, `NUMPY-LEARNING/`), not a runnable suite — don't treat it as one.

## Architecture

`src/main.py` is the whole app: an infinite loop wiring four modules together.

```
record() ─► transcribe(file) ─► decide_action(query) ─► tell_time()/set_timer() ─► say()
audio.audio   audio.stt          intent.intent           utils.utils                audio.tts
```

- **`audio/audio.py`** — `record()` blocks until space is pressed (busy-wait), captures 16-bit mono @ 44.1kHz via PyAudio, and always writes to a fixed `output.wav`.
- **`audio/stt.py`** — `transcribe()` reloads the Whisper `tiny` model on every call (no caching yet) and returns the text.
- **`intent/intent.py`** — `decide_action()` is a substring keyword matcher (`"what is the time"`, `"timer"`), not the DistilBERT/multi-signal router described in the design doc.
- **`audio/tts.py`** — `say()` uses a module-level `pyttsx3` engine (blocking, offline). The design doc calls for Piper TTS eventually.
- **`utils/utils.py`** — action handlers (`tell_time`, `set_timer`, `end_timer`).

### Import convention (important)

Modules import each other by top-level package name, e.g. `from audio.audio import record`, `from intent.intent import decide_action`. This means **`src/` must be the working directory / on `sys.path`** — run from inside `src/`, not from the repo root. Keep new imports in this same `package.module` form.
