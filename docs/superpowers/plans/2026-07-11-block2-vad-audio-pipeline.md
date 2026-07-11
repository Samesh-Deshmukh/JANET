# Block 2 — Always-Listening VAD & Audio Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace push-to-talk capture with a fully always-listening pipeline where Silero VAD detects speech start/end on a continuous mic stream, with a pre-roll buffer so the first word is never clipped.

**Architecture:** A continuous 16kHz frame generator (`audio.py`) feeds every 512-sample frame to both a pre-roll ring buffer (`ring_buffer.py`) and a Silero VAD wrapper (`vad.py`). `main.py` collects frames between a VAD `start` and `end` event — prepending the ring-buffer pre-roll — then runs the existing STT → dispatch → TTS chain. Single-threaded this block; threading is Block 5.

**Tech Stack:** Python 3.11, PyAudio (capture), NumPy, `silero-vad` (VAD), Whisper (existing STT), scipy (`resample_poly`, fallback path only).

**Design spec:** `docs/superpowers/specs/2026-07-10-block2-vad-audio-pipeline-design.md`

## Global Constraints

- **Run location:** all code runs with `src/` as cwd / on `sys.path`. Imports use `package.module` form (e.g. `from audio.vad import SpeechDetector`). Never run from repo root.
- **Local-first:** no cloud/remote calls. `silero-vad` downloads its model once on first run, then is fully offline — acceptable. Nothing else may reach the network.
- **No pytest / no test suite:** the repo has none and `tests/` is gitignored scratch. Tests are inline `if __name__ == "__main__":` self-check blocks with plain `assert`s, run by hand as `python audio/<module>.py`. Follow the `stt.py` pattern. Do NOT add pytest or create files under `tests/`.
- **One concern per file** (CLAUDE.md working style). New subsystems get their own module.
- **Understandable over clever:** simple, explainable code; comment intent, not mechanics. No heavy abstractions.
- **Frame contract:** Silero VAD at 16kHz requires **exactly 512-sample** float32 frames (verified against Silero docs). All frame sizing keys off `FRAME_SAMPLES = 512`.
- **Audio dtype:** float32 in range [-1, 1] at 16kHz mono everywhere downstream (what Whisper and Silero both want).
- **`*.wav` is gitignored:** never commit audio fixtures; the offline harness takes a path argument to a local file the user supplies.
- **No deletions this block:** `src/utils/hotkey.py` stays in the tree, just unimported.

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/audio/ring_buffer.py` | Create | Fixed-size float32 pre-roll ring buffer: `push(frame)`, `snapshot()`, `clear()`. Pure, mic-free. |
| `src/audio/vad.py` | Create | `SpeechDetector`: wraps Silero `VADIterator`; `process(frame) -> "start"|"end"|None`, `reset()`. Plus `segment_wav(path)` offline harness. |
| `src/audio/audio.py` | Rewrite | Continuous frame source `frames()`: persistent 16kHz PyAudio stream yielding 512-sample float32 frames forever. Drop SPACE-gated `record()`. |
| `src/main.py` | Modify | Listen-forever loop: frame → ring + VAD → collect utterance → STT → dispatch → TTS → reset. |
| `requirements.txt` | Modify | Add `silero-vad`. |
| `CLAUDE.md` | Modify | Update Architecture / Target stack / Running sections (workspace copy is gitignored; edit the JANET-repo `CLAUDE.md`). |

**Task order & dependencies:** Task 1 (ring buffer) and Task 2 (VAD) are independent and each self-contained. Task 3 (frame source) is independent of both. Task 4 (main loop) consumes all three. Task 5 (docs/deps housekeeping) closes out.

---

### Task 1: Pre-roll ring buffer

**Files:**
- Create: `src/audio/ring_buffer.py`

**Interfaces:**
- Consumes: nothing (pure NumPy).
- Produces:
  - `RingBuffer(capacity_samples: int)` — construct sized in samples.
  - `RingBuffer.push(frame: np.ndarray) -> None` — append a float32 frame, overwriting oldest samples once full.
  - `RingBuffer.snapshot() -> np.ndarray` — retained audio in chronological (oldest→newest) order, float32; length ≤ capacity.
  - `RingBuffer.clear() -> None` — empty the buffer.
  - Module constant `PRE_ROLL_MS = 500` and helper: at 16kHz, capacity = `PRE_ROLL_MS * 16` samples.

**Why:** When VAD says "speech started," the model needs ~60–90ms and the speaker's first phoneme has already passed. Prepending the last ~500ms of audio guarantees the first word isn't clipped.

- [ ] **Step 1: Write the module with its inline self-check**

Create `src/audio/ring_buffer.py`:

```python
"""Fixed-size pre-roll ring buffer for audio.

Silero VAD reports "speech started" a few frames after speech actually begins,
so by the time we start collecting, the first phoneme is already gone. This
buffer continuously retains the last PRE_ROLL_MS of audio; when speech starts we
prepend snapshot() to the utterance so the opening word survives.

Pure NumPy, no mic — run `python audio/ring_buffer.py` to exercise the asserts.
"""
import numpy as np

TARGET_RATE = 16000       # samples per second we operate at
PRE_ROLL_MS = 500         # how much audio to keep in front of a detected utterance
PRE_ROLL_SAMPLES = PRE_ROLL_MS * TARGET_RATE // 1000  # = 8000 samples


class RingBuffer:
    """Keeps the most recent `capacity` samples of float32 audio."""

    def __init__(self, capacity=PRE_ROLL_SAMPLES):
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._filled = 0     # how many valid samples we hold (caps at capacity)
        self._end = 0        # index one past the newest sample (write head)

    def push(self, frame):
        """Append a 1-D float32 frame, overwriting the oldest samples when full."""
        frame = np.asarray(frame, dtype=np.float32)
        n = len(frame)
        if n >= self.capacity:
            # frame alone fills/overflows the buffer: keep only its tail
            self._buf[:] = frame[-self.capacity:]
            self._end = 0
            self._filled = self.capacity
            return
        end = self._end
        first = min(n, self.capacity - end)   # part that fits before wrapping
        self._buf[end:end + first] = frame[:first]
        rest = n - first                      # part that wraps to the front
        if rest:
            self._buf[:rest] = frame[first:]
        self._end = (end + n) % self.capacity
        self._filled = min(self._filled + n, self.capacity)

    def snapshot(self):
        """Return retained audio, oldest -> newest, as a float32 array."""
        if self._filled < self.capacity:
            # not yet wrapped: valid data is [0 .. _end)
            return self._buf[:self._end].copy()
        # wrapped: oldest sample sits at _end, read from there around the ring
        return np.concatenate((self._buf[self._end:], self._buf[:self._end]))

    def clear(self):
        self._filled = 0
        self._end = 0


if __name__ == "__main__":
    # 1) Under-capacity: snapshot returns exactly what went in, in order.
    rb = RingBuffer(capacity=10)
    rb.push(np.array([1, 2, 3], dtype=np.float32))
    rb.push(np.array([4, 5], dtype=np.float32))
    assert np.array_equal(rb.snapshot(), [1, 2, 3, 4, 5]), rb.snapshot()

    # 2) Over-capacity across pushes: only the newest `capacity` samples remain.
    rb = RingBuffer(capacity=4)
    rb.push(np.array([1, 2, 3], dtype=np.float32))
    rb.push(np.array([4, 5, 6], dtype=np.float32))   # total 6 pushed, keep last 4
    assert np.array_equal(rb.snapshot(), [3, 4, 5, 6]), rb.snapshot()

    # 3) Single frame larger than capacity keeps its tail.
    rb = RingBuffer(capacity=3)
    rb.push(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    assert np.array_equal(rb.snapshot(), [3, 4, 5]), rb.snapshot()

    # 4) clear() empties it.
    rb.clear()
    assert rb.snapshot().size == 0

    # 5) Real sizing: 500ms at 16kHz is 8000 samples.
    assert PRE_ROLL_SAMPLES == 8000
    print("✅ ring_buffer self-checks passed")
```

- [ ] **Step 2: Run the self-check and verify it passes**

Run (from `src/`): `python audio/ring_buffer.py`
Expected: `✅ ring_buffer self-checks passed` and exit code 0. No `AssertionError`.

- [ ] **Step 3: Commit**

```bash
git add src/audio/ring_buffer.py
git commit -m "Add pre-roll ring buffer for VAD (Block 2)"
```

---

### Task 2: Silero VAD wrapper + offline segmentation harness

**Files:**
- Create: `src/audio/vad.py`

**Interfaces:**
- Consumes: `silero-vad` package (`from silero_vad import load_silero_vad, VADIterator, read_audio`).
- Produces:
  - `SpeechDetector()` — loads the Silero model once (module-level singleton, like `stt.py`).
  - `SpeechDetector.process(frame: np.ndarray) -> "start" | "end" | None` — feed one 512-sample float32 frame; returns a transition event or `None`.
  - `SpeechDetector.reset() -> None` — clear VAD state between utterances (delegates to `VADIterator.reset_states()`).
  - `segment_wav(path: str) -> list[tuple[float, float]]` — run a wav file through the same frame→VAD loop and return/print (start_s, end_s) speech segments.
  - Constants `FRAME_SAMPLES = 512`, `VAD_THRESHOLD = 0.5`, `MIN_SILENCE_MS = 600`, `SPEECH_PAD_MS = 100`.

**Why:** Silero is an LSTM trained on real speech — far fewer false triggers than energy thresholds. `VADIterator` is its streaming interface: feed fixed frames, get `{'start': t}` / `{'end': t}` dicts. `MIN_SILENCE_MS` is the knob that keeps JANET from cutting off the end of a sentence.

- [ ] **Step 1: Add `silero-vad` to requirements and install it**

Edit `requirements.txt` — under the STT section add:

```
silero-vad==5.1.2        # LSTM voice-activity detection; model caches locally after first run
```

Then install into the venv:

Run (from repo root, venv active): `pip install silero-vad==5.1.2`
Expected: installs `silero-vad` and its `onnxruntime`/`torch` deps (torch already present). If `5.1.2` is unavailable, use the newest `5.x` and record the pinned version in the commit message.

- [ ] **Step 2: Verify Silero imports and its frame-size contract**

Run (from `src/`):

```bash
python -c "from silero_vad import load_silero_vad, VADIterator; m=load_silero_vad(); print('silero ok')"
```

Expected: prints `silero ok` (downloads the ~1–2MB model on first run, then cached). Confirms the dependency and offline-after-first-run behavior before we build on it.

- [ ] **Step 3: Write `vad.py` with the wrapper and offline harness**

Create `src/audio/vad.py`:

```python
"""Silero voice-activity detection: is the user speaking right now?

Silero VAD is a small LSTM trained on real speech. Its streaming interface,
VADIterator, takes fixed 512-sample frames (at 16kHz) and returns a dict the
moment speech starts ({'start': seconds}) or ends ({'end': seconds}). We wrap
that into process(frame) -> "start" | "end" | None so main.py stays simple.

Run `python audio/vad.py <file.wav>` to print the speech segments it finds in a
recording — deterministic, no mic needed.
"""
import sys
import numpy as np
from silero_vad import load_silero_vad, VADIterator, read_audio

TARGET_RATE = 16000
FRAME_SAMPLES = 512      # Silero's required window at 16kHz — do not change
VAD_THRESHOLD = 0.5      # speech-probability cutoff (higher = stricter)
MIN_SILENCE_MS = 600     # trailing silence before "end" fires; stops early cut-off
SPEECH_PAD_MS = 100      # padding Silero adds around detected speech edges

_model = None


def _get_model():
    """Load the Silero model once and reuse it (same pattern as stt.py)."""
    global _model
    if _model is None:
        print("⏳ Loading Silero VAD model (first time only)...")
        _model = load_silero_vad()
        print("✅ Silero VAD ready.")
    return _model


class SpeechDetector:
    """Stateful streaming detector. Feed it one frame at a time."""

    def __init__(self):
        self._it = VADIterator(
            _get_model(),
            threshold=VAD_THRESHOLD,
            sampling_rate=TARGET_RATE,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )

    def process(self, frame):
        """Feed one 512-sample float32 frame. Returns 'start', 'end', or None."""
        import torch
        frame = np.asarray(frame, dtype=np.float32)
        out = self._it(torch.from_numpy(frame), return_seconds=True)
        if out is None:
            return None
        if "start" in out:
            return "start"
        if "end" in out:
            return "end"
        return None

    def reset(self):
        """Clear internal state so the next utterance starts fresh."""
        self._it.reset_states()


def segment_wav(path):
    """Run a wav file through the same frame->VAD loop; return [(start_s, end_s)].

    Deterministic offline check that segmentation works without a live mic.
    """
    wav = read_audio(path, sampling_rate=TARGET_RATE)   # float32 torch tensor @16k
    wav = wav.numpy()
    detector = SpeechDetector()
    segments, start = [], None
    for i in range(0, len(wav) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = wav[i:i + FRAME_SAMPLES]
        event = detector.process(frame)
        if event == "start":
            start = i / TARGET_RATE
        elif event == "end" and start is not None:
            segments.append((round(start, 2), round(i / TARGET_RATE, 2)))
            start = None
    if start is not None:                                # speech ran to EOF
        segments.append((round(start, 2), round(len(wav) / TARGET_RATE, 2)))
    return segments


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python audio/vad.py <file.wav>")
        raise SystemExit(2)
    segs = segment_wav(sys.argv[1])
    print(f"Detected {len(segs)} speech segment(s):")
    for s, e in segs:
        print(f"  {s:6.2f}s -> {e:6.2f}s  ({e - s:.2f}s)")
```

- [ ] **Step 4: Verify the harness runs against a recording**

Record or reuse a short wav with one spoken sentence surrounded by silence (any local file; `*.wav` stays uncommitted). Run (from `src/`):

```bash
python audio/vad.py /path/to/sample.wav
```

Expected: prints at least one segment whose start is at/after the silence and whose end is near where speech stops. A file of pure silence prints `Detected 0 speech segment(s):`. This proves `process()` and the start/end event mapping work end-to-end.

- [ ] **Step 5: Commit**

```bash
git add src/audio/vad.py requirements.txt
git commit -m "Add Silero VAD wrapper + offline segment_wav harness (Block 2)"
```

---

### Task 3: Continuous frame source

**Files:**
- Rewrite: `src/audio/audio.py`

**Interfaces:**
- Consumes: PyAudio, NumPy, scipy `resample_poly` (fallback path only).
- Produces:
  - `frames() -> Iterator[np.ndarray]` — generator yielding 512-sample float32 [-1,1] @16kHz frames forever from a persistent input stream.
  - Keeps `_bytes_to_array`, `_volume_bar`, `_resample_to_16k` helpers (reused/retained). Removes `record()` and the `is_pressed` import.
  - Module constant `FRAME_SAMPLES = 512` (must match `vad.py`).

**Why:** Always-listening needs one long-lived stream that never blocks on a key. Opening directly at 16kHz makes each read already Silero-sized with no resample; the 44.1kHz path is a fallback for hardware that refuses 16kHz.

- [ ] **Step 1: Rewrite `audio.py` as a frame generator**

Replace the entire contents of `src/audio/audio.py` with:

```python
"""Continuous microphone frame source for the always-listening pipeline.

Block 1 captured only while SPACE was held. Block 2 removes the key entirely:
frames() opens one long-lived 16kHz mono stream and yields fixed 512-sample
float32 frames forever. Silero VAD (audio/vad.py) decides when speech starts and
stops; main.py collects the frames in between.

We try to open the mic directly at 16kHz so each read is already Silero-sized. If
the hardware refuses 16kHz we fall back to 44.1kHz + polyphase resampling.
"""
import pyaudio
import numpy as np
from scipy.signal import resample_poly

FORMAT = pyaudio.paInt16
CHANNELS = 1
TARGET_RATE = 16000        # what Whisper and Silero want
FALLBACK_RATE = 44100      # used only if the mic won't open at 16kHz
FRAME_SAMPLES = 512        # Silero's required window at 16kHz (matches vad.py)
MAX_INT16 = 32768.0        # normalize int16 -> float32 [-1, 1]


def _bytes_to_array(data):
    """Raw PyAudio int16 bytes -> numpy int16 array. Pure, testable without a mic."""
    return np.frombuffer(data, dtype=np.int16)


def _volume_bar(chunk, width=40):
    """RMS of one frame -> a text meter, for optional --meter debugging."""
    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
    level = min(int(rms / MAX_INT16 * width * 4), width)
    return '[' + '#' * level + '-' * (width - level) + ']'


def _resample_to_16k(audio, src_rate):
    """Downsample float32 audio from src_rate to 16kHz (polyphase anti-aliasing)."""
    if src_rate == TARGET_RATE:
        return audio
    return resample_poly(audio, TARGET_RATE, src_rate).astype(np.float32)


def _open_stream(p):
    """Open the mic at 16kHz if possible, else 44.1kHz. Returns (stream, rate, read_size)."""
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=TARGET_RATE,
                        input=True, frames_per_buffer=FRAME_SAMPLES)
        return stream, TARGET_RATE, FRAME_SAMPLES
    except Exception:
        # Hardware refused 16kHz: capture at 44.1k and resample. Read a chunk that
        # resamples to ~FRAME_SAMPLES (441 in -> 160 out per 16k frame => 512*441//160).
        read_size = FRAME_SAMPLES * FALLBACK_RATE // TARGET_RATE
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=FALLBACK_RATE,
                        input=True, frames_per_buffer=read_size)
        return stream, FALLBACK_RATE, read_size


def frames(meter=False):
    """Yield 512-sample float32 [-1,1] @16kHz frames forever from the mic.

    At 16kHz each read is already 512 samples. On the 44.1kHz fallback we
    resample each read and re-slice to exact 512-sample frames so the VAD
    contract holds regardless of hardware.
    """
    p = pyaudio.PyAudio()
    stream, rate, read_size = _open_stream(p)
    print(f"🎤 Listening continuously at {rate} Hz. (Ctrl-C to quit.)")
    leftover = np.zeros(0, dtype=np.float32)
    try:
        while True:
            raw = _bytes_to_array(stream.read(read_size, exception_on_overflow=False))
            if meter:
                print(f"\r  {_volume_bar(raw)}", end='', flush=True)
            audio = raw.astype(np.float32) / MAX_INT16
            audio = _resample_to_16k(audio, rate)
            # Re-slice into exact 512-sample frames (needed on the fallback path;
            # a no-op remainder at 16kHz where read_size already == FRAME_SAMPLES).
            audio = np.concatenate((leftover, audio))
            n_full = len(audio) // FRAME_SAMPLES
            for k in range(n_full):
                yield audio[k * FRAME_SAMPLES:(k + 1) * FRAME_SAMPLES]
            leftover = audio[n_full * FRAME_SAMPLES:]
    finally:
        stream.close()
        p.terminate()
```

- [ ] **Step 2: Verify frames() yields correctly-shaped frames from the real mic**

Run (from `src/`):

```bash
python -c "from audio.audio import frames, FRAME_SAMPLES; g=frames(); \
fr=[next(g) for _ in range(5)]; \
import numpy as np; \
assert all(f.shape==(FRAME_SAMPLES,) for f in fr), [f.shape for f in fr]; \
assert all(f.dtype==np.float32 for f in fr); \
print('✅ frames() yields 5x', fr[0].shape, fr[0].dtype)"
```

Expected: prints `✅ frames() yields 5x (512,) float32`. Speak or tap the mic while it runs — this confirms the stream opens, the rate negotiation works, and frames are Silero-sized. (If it hangs with no output, the mic isn't readable as your user — see CLAUDE.md "Running" notes.)

- [ ] **Step 3: Commit**

```bash
git add src/audio/audio.py
git commit -m "Replace push-to-talk record() with continuous frames() source (Block 2)"
```

---

### Task 4: Always-listening main loop

**Files:**
- Modify: `src/main.py`

**Interfaces:**
- Consumes: `frames()` (Task 3), `SpeechDetector` (Task 2), `RingBuffer`/`PRE_ROLL_SAMPLES` (Task 1), existing `transcribe`, `respond`, `say`, `Context`.
- Produces: an always-on `main()` — no `record()`, no SPACE.
- Constant `MAX_UTTERANCE_S = 30` (hard cap so a stuck-open VAD can't collect forever).

**Why:** This is the block's payoff — wiring the frame source, pre-roll, and VAD into the existing STT→dispatch→TTS chain so JANET responds without a key.

- [ ] **Step 1: Rewrite `main.py` as the listen-forever loop**

Replace the entire contents of `src/main.py` with:

```python
"""JANET always-listening entrypoint.

Every mic frame is (a) pushed into a pre-roll ring buffer and (b) fed to Silero
VAD. When VAD reports speech start we begin an utterance, prepending the ring
buffer so the first word isn't clipped; when it reports end (or we hit the length
cap) we transcribe, dispatch, and speak, then reset and keep listening.

Single-threaded for Block 2: capture pauses during STT/TTS, so JANET doesn't hear
itself. Concurrency comes in Block 5. Run from src/:  python main.py
"""
import numpy as np

from audio.audio import frames, FRAME_SAMPLES, TARGET_RATE
from audio.vad import SpeechDetector
from audio.ring_buffer import RingBuffer, PRE_ROLL_SAMPLES
from audio.stt import transcribe
from audio.tts import say
from intent.dispatch import respond
from utils.context import Context

MAX_UTTERANCE_S = 30
MAX_UTTERANCE_FRAMES = MAX_UTTERANCE_S * TARGET_RATE // FRAME_SAMPLES


def _handle(utterance):
    """Transcribe one collected utterance and respond to it."""
    audio = np.concatenate(utterance)
    query = transcribe(audio)
    print(f"🗣  You said: {query.strip()}")
    if not query.strip():
        return                      # Whisper heard nothing intelligible; stay quiet
    ctx = Context(speak=say, query=query)
    reply = respond(query, ctx)
    print(f"⚙️  Reply: {reply}")
    say(reply)


def main():
    print("JANET is running (always-listening). Press Ctrl-C to quit.")
    detector = SpeechDetector()
    ring = RingBuffer(capacity=PRE_ROLL_SAMPLES)
    utterance = None                # None = idle; a list = actively collecting

    for frame in frames():
        ring.push(frame)
        event = detector.process(frame)

        if utterance is None:
            if event == "start":
                # seed with the pre-roll so the opening word survives
                utterance = [ring.snapshot()]
        else:
            utterance.append(frame)
            if event == "end" or len(utterance) >= MAX_UTTERANCE_FRAMES:
                _handle(utterance)
                detector.reset()
                ring.clear()
                utterance = None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 JANET stopped.")
```

- [ ] **Step 2: Live acceptance run**

Run (from `src/`, as your normal user — NOT sudo): `python main.py`

Speak a known intent after a beat of silence, e.g. "what time is it", pause, then try "set a timer for ten seconds". Expected:
- No SPACE press needed; it reacts on its own after you stop talking.
- `🗣  You said:` shows the full sentence with the **first word intact** (pre-roll working).
- The sentence isn't **cut off at the end** (MIN_SILENCE_MS working).
- Sitting silent produces no spurious `You said:` lines.
- Ctrl-C prints `👋 JANET stopped.` cleanly.

If the first word is clipped, raise `PRE_ROLL_MS` in `ring_buffer.py`; if endings cut off, raise `MIN_SILENCE_MS` in `vad.py`. Re-run until the four bullets hold.

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "Wire always-listening VAD loop into main (Block 2)"
```

---

### Task 5: Documentation & dependency housekeeping

**Files:**
- Modify: `CLAUDE.md` (the JANET-repo copy; it is gitignored in the workspace but tracked in JANET's own repo — edit it there)
- Modify: `requirements.txt` (prune legacy push-to-talk deps if no longer imported)

**Interfaces:** none (docs/config only).

**Why:** CLAUDE.md's "Keep this file current" rule requires the map to match the code in the same change: push-to-talk is gone, three audio modules exist, Silero is in the stack.

- [ ] **Step 1: Update the requirements comments for the input libs**

In `requirements.txt`, the `keyboard`/`pynput` entries are already flagged legacy. Add a one-line note that push-to-talk (and thus `evdev`/`hotkey.py`) is now dormant as of Block 2 — do NOT remove `evdev` (still used by the retained `hotkey.py`), just annotate:

```
# NOTE: as of Block 2 (always-listening VAD) capture no longer uses push-to-talk.
# keyboard/pynput remain legacy/unused; evdev + utils/hotkey.py are kept but dormant.
```

- [ ] **Step 2: Update `CLAUDE.md` Architecture / Target stack / Running sections**

Make these edits to `CLAUDE.md`:

- **Target stack:** change the VAD bullet from "Not yet in the code" to: "**VAD:** Silero VAD, streaming via `VADIterator` (512-sample frames @16kHz). Live as of Block 2." Update the pipeline line to note it is always-listening (no wake word, no push-to-talk).
- **Running:** replace the push-to-talk paragraph ("hold space to record") with the always-listening behavior: "JANET listens continuously; Silero VAD detects when you start and stop speaking — no key press. Run `cd src && python main.py` as your normal user. Ctrl-C to quit." Note the `input`-group / non-root requirement still applies because `hotkey.py` remains in the tree but add that it is no longer on the run path.
- **Architecture:** update the flow diagram and module list: `frames()` (continuous source) → `RingBuffer` (pre-roll) + `SpeechDetector` (Silero) → collected utterance → `transcribe` → `respond` → `say`. Add `audio/vad.py` and `audio/ring_buffer.py` to the module descriptions; mark `record()`/SPACE and `utils/hotkey.py` as removed-from-path/dormant.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt CLAUDE.md
git commit -m "Docs: always-listening pipeline; mark push-to-talk dormant (Block 2)"
```

---

## Self-Review Notes

- **Spec coverage:** always-listening loop (Task 4) ✓; Silero via `SpeechDetector` (Task 2) ✓; pre-roll ring buffer (Task 1) ✓; 16kHz-direct with 44.1k fallback frame source (Task 3) ✓; `silero-vad` dep, offline-after-first-run (Task 2 step 1–2) ✓; `segment_wav` offline harness (Task 2) ✓; single-threaded, mic deaf during STT/TTS (Task 4 docstring) ✓; 30s utterance cap (Task 4) ✓; `hotkey.py` retained not deleted (Task 5) ✓; CLAUDE.md sections updated (Task 5) ✓. Deferred items (threading, adaptive threshold, sentence-completion classifier, source classification) are explicitly out of scope and have no task, as intended.
- **Constant consistency:** `FRAME_SAMPLES = 512` defined in `vad.py` and `audio.py`, imported (not redefined) by `main.py`; `PRE_ROLL_SAMPLES` defined in `ring_buffer.py`, imported by `main.py`; `TARGET_RATE = 16000` consistent across `audio.py`, `vad.py`, `ring_buffer.py`. `process()` returns exactly `"start"|"end"|None` and every consumer branches on those three.
- **No pytest:** all tests are inline `__main__` self-checks or the `segment_wav` harness, per the repo convention and the user's explicit choice.
