# JANET

**Just Another Neural Execution Tool** — a fully-local, privacy-first, wake-word-free voice assistant.

Everything runs on-device. No cloud calls for core functions; no wake word to press or say — JANET listens continuously and works out for itself when you're talking to it.

> **Status: early prototype.** The audio pipeline and the intent brain (addressing scorer + trained classifier) work end-to-end. Most *action handlers* aren't built yet, so recognized intents like weather or email currently answer "I can't help with that yet." This is also a personal learning project — the code favours being understandable over clever.

## The pipeline

```
Mic ─► VAD ─► Whisper ─► normalize ─► Scorer ─► Classifier ─► Action ─► TTS
      Silero   (STT)                  (Layer 1)  (Layer 2)     handler   Piper*
```

Nothing is spoken unless **both** gates agree the utterance is a real request *to JANET*:

1. **Scorer** (`intent/scorer.py`) — "Was this addressed to me?" Scores linguistic signals (question/command shape, "janet", keywords…) against a threshold. Cheap, runs first.
2. **Classifier** (`intent/classifier.py`) — "What do they want?" A fine-tuned DistilBERT (13 intents, ~93% val accuracy). Also vetoes anything it reads as `NONE` (not a real intent) or is unsure about.

If either gate says no, JANET stays silent — which matters a lot for an always-listening mic.

## Current capabilities

| Works today | Recognized, handler not built yet |
|-------------|-----------------------------------|
| TIME, DATE, TIMER, CALC (digit math) | WEATHER, EMAIL, CALENDAR, REMINDER, ALARM, SMART_HOME, SYSTEM, GENERAL |

## Getting started

Requires audio hardware (mic + speaker) and **Python 3.11**. Run as your **normal user** (not `sudo` — a per-user PipeWire mic is unreachable as root).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src && python main.py          # runs from src/ — see the import note below
```

Then just talk. Silero VAD detects when you start and stop speaking (a short pre-roll buffer keeps your first word). Each utterance is transcribed, run through the two gates, and — if it's for JANET — acted on and spoken. The console shows the decision on every utterance:

```
🗣  You said: what time is it
🛡  Score: 65 (question +40, keyword +25) → addressed
🧠 Intent: TIME (89%)
⚙️  Reply: The time is 07:51 PM
```

`Ctrl-C` to quit.

> **Import convention:** modules import by top-level package (`from intent.scorer import score`), so `src/` must be the working directory — run from inside `src/`, not the repo root.

## Training the intent classifier

The classifier is fine-tuned locally (a GPU helps but isn't required):

```bash
cd src && python -m intent.train
```

This reads the labelled dataset in `data/text/{train,val}/`, fine-tunes `distilbert-base-uncased`, prints per-class metrics + a confusion matrix, and saves the model to `data/models/intent-distilbert/` (gitignored). Validate the dataset's format with `python data/text/validate.py`.

## Project layout

```
src/
  main.py            always-listening loop
  audio/             frames() source, Silero VAD, ring buffer, Whisper STT, TTS
  intent/            normalize · scorer (Layer 1) · classifier + train/dataset (Layer 2) · dispatch
  actions/           per-intent handlers (TIME, DATE, TIMER, CALC so far)
  utils/             context + helpers
data/
  text/              intent dataset (train/val), labels.txt, validate.py
  models/            trained model (gitignored)
```

## Principles

- **Local-first.** No cloud dependency for any core function — that's the whole point.
- **Understandable over clever.** Small, reviewable changes; the owner wants to explain every line.
- **Human-in-the-loop.** Nothing acts on you unless both gates agree it was meant for JANET.

## Target stack

Whisper (STT) · Silero VAD · DistilBERT intent classifier · a multi-signal addressing scorer · Piper TTS and a local Ollama/Gemma LLM fallback (both planned). Everything on-device.
