# Block 2 — Always-Listening VAD & Audio Pipeline

**Date:** 2026-07-10
**Status:** Design approved, ready for implementation plan
**Roadmap ref:** `Notes/JANET-Plan-this-summer-2026.txt` → Block 2 (weeks 4–5)
**Spec ref:** `Notes/JANET_v2.0.md` §2.5 (Voice Activity Detection)

## Goal

Replace the push-to-talk (`hold SPACE`) capture loop with a **fully
always-listening pipeline**: the mic runs continuously, Silero VAD decides on
its own when speech starts and stops, and JANET transcribes and responds
without any key press. This is the core of the "no wake word" vision.

By the end: JANET reliably captures everything you say — without clipping the
first word or cutting off the end of a sentence — and ignores silence.

## Scope decision (approved)

The roadmap prescribes building a naive energy+ZCR VAD first, *then* swapping in
Silero. **We are skipping the hand-rolled detector and going straight to
Silero** (option "C"). This trades away the "feel why Silero is better" learning
step in exchange for reaching the always-listening target faster. Recorded here
so the deviation from the plan is intentional and visible.

Also **in scope but deferred to later blocks** (do NOT build here):
- Threading (Block 5). Block 2 stays **single-threaded**: the mic is naturally
  deaf while Whisper runs and while JANET speaks. Accepted trade-off.
- Adaptive VAD threshold by noise floor (spec §2.5 table) — fixed threshold now.
- Sentence-completion classifier for the continuation window (spec §2.5) — we
  use Silero's built-in trailing-silence timeout instead.
- Audio source classification, diarization, AGC, multi-mic (spec §2.6–2.7).

## Architecture

New module split (honoring "one concern per file"):

| File | Concern | Public surface |
|------|---------|----------------|
| `src/audio/vad.py` | "Did speech start or stop?" | `SpeechDetector` wrapping Silero `VADIterator` |
| `src/audio/ring_buffer.py` | "Keep the last N seconds of audio" | fixed-size NumPy ring buffer + read-back |
| `src/audio/audio.py` | Continuous frame source | frame generator over a persistent PyAudio stream |
| `src/main.py` | Orchestration | listen-forever loop |
| `src/utils/hotkey.py`* | (dormant) | kept in tree, no longer imported |

\* `hotkey.py` stays where it is, unimported. No deletions this block.

### Data flow

```
continuous 16kHz frames (512 samples / 32ms each)
        │
        ├──► ring_buffer.push(frame)        # always: last ~0.5s retained
        │
        └──► SpeechDetector.process(frame) ─► event: START | END | None
                    │
        on START:  begin utterance, prepend ring-buffer pre-roll
        collecting: append every frame
        on END:    concatenate → transcribe → respond → say → reset → resume
```

## Component detail

### `SpeechDetector` (`vad.py`)

- Thin wrapper over Silero VAD's `VADIterator`. Loads the model once (module-level
  singleton, same pattern as `stt.py`'s Whisper cache).
- `process(frame) -> "start" | "end" | None` — feeds one 512-sample float32 frame,
  returns a transition event or `None`.
- `reset()` — clears VAD internal state between utterances (delegates to Silero's
  `VADIterator.reset_states()`).
- Tunables as named constants at the top of the file:
  - `VAD_THRESHOLD` (default 0.5) — speech probability cutoff.
  - `MIN_SILENCE_MS` (default ~600) — trailing silence before END fires. This is
    the knob that stops JANET cutting off the end of a sentence.
  - `SPEECH_PAD_MS` — Silero's own edge padding.

### `RingBuffer` (`ring_buffer.py`)

- Fixed-capacity NumPy float32 ring buffer sized to `PRE_ROLL_MS` (default ~500ms).
- `push(frame)` — append, overwriting oldest.
- `snapshot() -> np.ndarray` — return retained audio in chronological order, to
  prepend when speech starts so the first word isn't lost.
- Pure and mic-free → unit-testable directly.

### Frame source (`audio.py`)

- Replace SPACE-gated `record()` with `frames()`: open PyAudio **once** and yield
  fixed 512-sample float32 frames forever.
- **Sample rate:** attempt to open the stream **directly at 16kHz mono** so frames
  are already Silero-sized with no per-chunk resampling (PipeWire resamples for us
  now that we run as a normal user). **Fallback:** open at 44.1kHz and resample
  each frame to 16kHz via the existing `resample_poly` path only if the 16kHz open
  fails. Which path works is verified on this machine as step 1 of implementation.
- Keep the RMS volume meter as an optional debug print; it must not spam the
  normal run.

### Loop (`main.py`)

```
detector = SpeechDetector()
ring = RingBuffer()
for frame in frames():
    ring.push(frame)
    event = detector.process(frame)
    if event == "start":
        utterance = [ring.snapshot()]
    elif collecting:
        utterance.append(frame)
        if event == "end" or too_long(utterance):   # cap ~30s
            audio = concatenate(utterance)
            query = transcribe(audio)
            reply = respond(query, Context(speak=say, query=query))
            say(reply)
            detector.reset()
```

Single-threaded: capture pauses during STT/dispatch/TTS. JANET won't hear itself
speak. Latency here is knowingly accepted and resolved in Block 5.

## Config & dependencies

- Add `silero-vad` to `requirements.txt`. The model downloads once on first run,
  then is cached locally — **fully offline thereafter** (local-first preserved).
- All VAD/buffer tunables are named constants at the top of their modules, not
  scattered magic numbers.

## Testing strategy

Always-listening is hard to test by hand, so keep a deterministic offline path:

- `segment_wav(path)` helper runs a recorded `.wav` through the **exact same**
  frame → VAD loop and prints the detected speech segments (start/end times).
  Validates segmentation without a live mic.
- `RingBuffer` gets direct unit coverage (push/snapshot/wraparound) since it's pure.
- Manual acceptance on real hardware: speak several utterances with natural
  pauses; confirm no clipped first words, no early cut-offs, silence ignored.

## Definition of done

- `main.py` runs with no SPACE key; speaking triggers transcription + response.
- First word is never clipped (pre-roll works); sentences aren't cut off early
  (trailing-silence timeout works); silence produces no spurious utterances.
- `segment_wav` reproduces segmentation offline.
- `requirements.txt` updated; still runs fully offline after first model fetch.
- `CLAUDE.md` "Architecture", "Target stack", and "Running" sections updated to
  reflect always-listening (no push-to-talk), the new modules, and Silero.

## Out of scope (explicit)

Threading, adaptive thresholds, sentence-completion classifier, source
classification, diarization, AGC, multi-mic, Piper TTS, DistilBERT intent. Those
belong to Blocks 3–5 and later spec sections.
