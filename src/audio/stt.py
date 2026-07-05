import whisper

_model = None


def _get_model():
    """Load the Whisper model once and reuse it — reloading on every call was the real slow path."""
    global _model
    if _model is None:
        print("⏳ Loading Whisper 'tiny' model (first time only)...")
        _model = whisper.load_model("tiny")
        print("✅ Whisper model ready.")
    return _model


def transcribe(audio):
    # `audio` is a float32 numpy array at 16 kHz (from record()); Whisper also accepts a file path.
    print("📝 Transcribing...")
    result = _get_model().transcribe(audio)
    return result["text"]


if __name__ == "__main__":
    print(transcribe("output.wav"))
