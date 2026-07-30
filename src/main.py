"""JANET always-listening entrypoint.

Every mic frame is (a) pushed into a pre-roll ring buffer and (b) fed to Silero
VAD. When VAD reports speech start we begin an utterance, prepending the ring
buffer so the first word isn't clipped; when it reports end (or we hit the length
cap) we transcribe, dispatch, and speak, then reset and keep listening.

Single-threaded for Block 2: capture pauses during STT/TTS, so JANET doesn't hear
itself. Concurrency comes in Block 5. Run from src/:  python main.py
"""
import os
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from audio.audio import frames, FRAME_SAMPLES, TARGET_RATE
from audio.vad import SpeechDetector
from audio.ring_buffer import RingBuffer, PRE_ROLL_SAMPLES
from audio.stt import transcribe
from audio.tts import say
from intent.dispatch import respond
from utils.context import Context
from utils.history import ConversationHistory

MAX_UTTERANCE_S = 30
MAX_UTTERANCE_FRAMES = MAX_UTTERANCE_S * TARGET_RATE // FRAME_SAMPLES

# JANET_DEBUG_AUDIO=1 saves every captured utterance as a WAV.
#
# Worth having because transcripts alone can't tell two very different problems
# apart: "the microphone only caught a fragment" (a VAD or level problem) and
# "Whisper mangled a perfectly good recording" (a model problem). The fix for
# each is in a different file, so guessing wastes time — listen to the audio and
# you know immediately.
DEBUG_AUDIO = os.environ.get("JANET_DEBUG_AUDIO", "").lower() in ("1", "true", "yes")
DEBUG_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "debug_audio"


def _save_debug_audio(audio, transcript):
    """Write the utterance to a WAV named after what Whisper made of it."""
    try:
        DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "-" for c in transcript.strip())[:40]
        path = DEBUG_AUDIO_DIR / f"{datetime.now():%H%M%S}-{slug or 'silence'}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)                      # 16-bit
            handle.setframerate(TARGET_RATE)
            # frames() yields float32 in [-1, 1]; WAV wants int16.
            handle.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        print(f"💾 {len(audio) / TARGET_RATE:.1f}s of audio -> {path.name}")
    except OSError as exc:                              # never break the assistant
        print(f"⚠  couldn't save debug audio: {exc}")


def _handle(utterance, history):
    """Transcribe one collected utterance and respond to it."""
    audio = np.concatenate(utterance)
    query = transcribe(audio)
    print(f"🗣  You said: {query.strip()}")
    if DEBUG_AUDIO:
        _save_debug_audio(audio, query)
    if not query.strip():
        return                      # Whisper heard nothing intelligible; stay quiet
    ctx = Context(speak=say, query=query, history=history)
    reply = respond(query, ctx)
    if reply is None:
        return                          # not addressed / not for JANET — stay silent
    print(f"⚙️  Reply: {reply.text}")
    if reply.reasoning:
        print(f"💭 Why: {reply.reasoning}")
    say(reply.text)
    # Remember this addressed exchange so later questions have context — including
    # what the action returned and why JANET answered that way.
    history.add(query, reply.text, facts=reply.facts, reasoning=reply.reasoning)


def main():
    print("JANET is running (always-listening). Press Ctrl-C to quit.")
    detector = SpeechDetector()
    ring = RingBuffer(capacity=PRE_ROLL_SAMPLES)
    history = ConversationHistory()     # short-term memory, shared across turns
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
                _handle(utterance, history)
                detector.reset()
                ring.clear()
                utterance = None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 JANET stopped.")
