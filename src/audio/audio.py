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
