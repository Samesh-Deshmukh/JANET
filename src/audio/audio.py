import pyaudio
import numpy as np
from scipy.signal import resample_poly
import time

from utils.hotkey import is_pressed  # evdev-based SPACE detection; runs as normal user (no root)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
CAPTURE_RATE = 44100  # the rate the mic hardware actually gives us (works even under bare ALSA / sudo)
TARGET_RATE = 16000   # the rate Whisper wants — we resample down to it before transcribing
MAX_INT16 = 32768.0   # scale to normalize int16 samples into float32 [-1, 1]


def _bytes_to_array(data):
    """Raw PyAudio int16 bytes -> numpy int16 array. Pure, so it's testable without a mic."""
    return np.frombuffer(data, dtype=np.int16)


def _volume_bar(chunk, width=40):
    """RMS of one chunk -> a text meter, so you can watch your voice register while recording."""
    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
    # *4 because speech rarely comes close to full-scale int16; clamp to the bar width
    level = min(int(rms / MAX_INT16 * width * 4), width)
    return '[' + '#' * level + '-' * (width - level) + ']'


def _resample_to_16k(audio):
    """Downsample float32 audio from CAPTURE_RATE to 16 kHz (polyphase filter = proper anti-aliasing)."""
    if CAPTURE_RATE == TARGET_RATE:
        return audio
    # resample_poly reduces the ratio internally (gcd), so 44100->16000 becomes 160/441
    return resample_poly(audio, TARGET_RATE, CAPTURE_RATE).astype(np.float32)


def record():
    print("\n🎤 Ready — hold SPACE to talk, release to stop.")
    while not is_pressed():
        time.sleep(0.01)  # small sleep so waiting doesn't peg a CPU core

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=CAPTURE_RATE, input=True, frames_per_buffer=CHUNK)

    print("● Recording... (release SPACE to stop)")
    chunks = []
    while is_pressed():
        # exception_on_overflow=False: if we drain a hair too slowly, drop samples instead of crashing
        chunk = _bytes_to_array(stream.read(CHUNK, exception_on_overflow=False))
        chunks.append(chunk)
        print(f"\r  {_volume_bar(chunk)}", end='', flush=True)
    print()  # end the volume-meter line

    stream.close()
    p.terminate()

    if not chunks:
        print("■ No audio captured (space released too fast?).")
        return np.zeros(0, dtype=np.float32)

    # Collect chunks, then concatenate once — the Block 1 buffer. A true ring buffer comes in Block 2.
    audio = np.concatenate(chunks).astype(np.float32) / MAX_INT16
    audio = _resample_to_16k(audio)
    print(f"■ Stopped — captured {len(audio) / TARGET_RATE:.1f}s.")
    return audio
