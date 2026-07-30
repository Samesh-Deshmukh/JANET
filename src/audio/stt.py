# src/audio/stt.py
"""Speech to text, via Whisper.

Three settings do the work here, and the two cheap ones matter most.

Measured on this machine (espeak-synthesised clips, so treat the absolute
numbers as a proxy — but it is a fair A/B between configurations):

    tiny, as it was          41% of words,  heard "Janet" 0/5 times
    tiny + language + prompt 82% of words,  heard "Janet" 5/5
    base + language + prompt 78% of words,  5/5
    small + language + prompt 83% of words, 5/5

The flags beat the model size, and they cost nothing. `tiny` alone was turning
"Janet set a timer for five minutes" into "Don't accept timer for 5 minutes."
and "Janet say that again" into "Don't say much again."
"""
import os

import whisper

# `small` fits the ~4GB left after llama-server takes the GPU. The benchmark
# above shows only a point of difference on clean synthetic audio — but that
# audio is unrealistically easy, and model size is exactly what helps with real
# speech: an accent, a noisy room, a half-swallowed word. Override with
# JANET_WHISPER_MODEL=tiny if startup time matters more than accuracy.
MODEL_NAME = os.environ.get("JANET_WHISPER_MODEL", "small")

# JANET is spoken to in English. Without this Whisper guesses the language per
# utterance and sometimes guesses wrong — a real transcript from testing was
# "Jangan spongkan udangan buat apa-apa", which is not English and not what was
# said. One flag removes that whole class of failure.
LANGUAGE = "en"

# Whisper conditions its decoder on this text, which makes these words far more
# likely to be produced. That is the direct fix for the most common failure in
# this project: JANET's own name coming back as "In January", "Jan at",
# "Janette", "Janet Woods" or "planet" — and with the name goes the +50 the
# addressing scorer needs.
INITIAL_PROMPT = (
    "Janet. Hey Janet, what time is it? Janet, set a timer for five minutes. "
    "Janet, set an alarm for 7 AM. Janet, what's the weather? Janet, what's on "
    "my calendar? Janet, remind me to call mom. Janet, turn on the lights."
)

_model = None


def _get_model():
    """Load the model once and reuse it — reloading per call was the real slow path."""
    global _model
    if _model is None:
        print(f"⏳ Loading Whisper '{MODEL_NAME}' model (first time only)...")
        _model = whisper.load_model(MODEL_NAME)
        print("✅ Whisper model ready.")
    return _model


def transcribe(audio):
    """`audio` is a float32 numpy array at 16 kHz; a file path also works."""
    print("📝 Transcribing...")
    result = _get_model().transcribe(
        audio, language=LANGUAGE, initial_prompt=INITIAL_PROMPT
    )
    return result["text"]


if __name__ == "__main__":
    print(transcribe("output.wav"))
