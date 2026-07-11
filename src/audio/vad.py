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
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, VADIterator

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


def _read_wav_16k(path):
    """Load a wav file as float32 samples at TARGET_RATE.

    silero_vad ships its own read_audio(), but it shells out to
    torchaudio's sox/torchcodec backends, which this repo's pinned
    torchaudio build (matched to the CUDA 12.8 / RTX 5060 Ti torch wheel)
    doesn't have — torchaudio.list_audio_backends() no longer exists there.
    So we read the wav ourselves with scipy, the same library audio.py
    already uses for the mic's 44.1k -> 16k resample.
    """
    rate, data = wavfile.read(path)
    # Normalize dtype FIRST, on the raw array: for a stereo int wav, averaging
    # channels first would upcast to float64 and skip the integer branch, leaking
    # ±32767 magnitudes through. So scale to [-1, 1] here, then downmix.
    if np.issubdtype(data.dtype, np.integer):
        # int16 divides by 32768.0 to match audio.py's MAX_INT16 convention;
        # other int widths use their own full-scale value.
        full_scale = 32768.0 if data.dtype == np.int16 else float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / full_scale
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:              # stereo -> mono by averaging channels
        data = data.mean(axis=1).astype(np.float32)
    if rate != TARGET_RATE:
        data = resample_poly(data, TARGET_RATE, rate).astype(np.float32)
    return data


def segment_wav(path):
    """Run a wav file through the same frame->VAD loop; return [(start_s, end_s)].

    Deterministic offline check that segmentation works without a live mic.
    """
    wav = _read_wav_16k(path)
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
