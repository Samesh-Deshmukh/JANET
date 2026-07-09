import subprocess
import tempfile
import os

# Local, offline text-to-speech.
#
# We synthesize speech with espeak-ng (writing a temp WAV) and play it with
# `paplay`. Why not pyttsx3? pyttsx3's espeak driver shells out to `aplay`
# (ALSA), which isn't installed on this PipeWire system, so playback failed
# silently ("aplay: command not found"). `paplay` ships with the PipeWire
# stack and talks to it natively, so no ALSA compatibility shim is needed.
#
# Both `espeak-ng` and `paplay` are simple command-line tools, which keeps this
# easy to follow. Slated to be replaced by Piper later (see the summer plan).


def say(phrase):
    """Speak `phrase` out loud, blocking until playback finishes."""
    print("Said :", phrase)

    # espeak-ng needs a file to write to, so use a throwaway temp WAV.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        # 1. Text -> WAV on disk.
        subprocess.run(["espeak-ng", phrase, "-w", wav_path], check=True)
        # 2. WAV -> speakers via PipeWire.
        subprocess.run(["paplay", wav_path], check=True)
    finally:
        # Always clean up the temp file, even if playback errored.
        os.unlink(wav_path)
