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
