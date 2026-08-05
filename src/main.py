"""JANET always-listening entrypoint.

Every mic frame is (a) pushed into a pre-roll ring buffer and (b) fed to Silero
VAD. When VAD reports speech start we begin an utterance, prepending the ring
buffer so the first word isn't clipped; when it reports end (or we hit the length
cap) the utterance is handed off to be transcribed, dispatched and spoken.

**Block 5: three threads.** Capture, thinking and speaking are separate, so the
microphone keeps running while JANET works:

    [capture thread]  frames -> ring + VAD -> utterance -> _utterances queue
    [main thread]     utterance -> Whisper -> respond() -> speaker.speak()
    [speech thread]   audio/speaker.py -> espeak -> paplay

Why it mattered, measured on this machine: Whisper takes 0.10s and the language
model ~2s, but **playing a typical answer takes 8.5 seconds** — and the old
single-threaded loop was deaf for all of it. Anything said to JANET while it was
talking, or in the second after, was simply never heard.

JANET still doesn't listen to *itself*: there's no echo cancellation by default,
so the capture thread discards frames while `speaker.is_speaking()`. See
"Barge-in" in the README for turning that on. Run from src/:  python main.py
"""
import os
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from audio.audio import frames, FRAME_SAMPLES, TARGET_RATE
from audio.vad import SpeechDetector
from audio.ring_buffer import RingBuffer, PRE_ROLL_SAMPLES
from audio.stt import transcribe
from audio import speaker
from actions import alarm_action, timer_action, reminder_action
from intent.dispatch import respond
from utils.context import Context
from utils.history import ConversationHistory

# Lowered from 30s in Block 5. Continuous background speech — a video, a
# podcast, two people talking — never gives the VAD the silence it needs to end
# an utterance, so it runs to this cap. At 30s a real command spoken over a
# playing video waited half a minute to be answered (observed live). Nobody
# speaks a 15-second command, so cutting here costs nothing and halves the
# worst case.
MAX_UTTERANCE_S = 15
MAX_UTTERANCE_FRAMES = MAX_UTTERANCE_S * TARGET_RATE // FRAME_SAMPLES

# Utterances waiting to be transcribed. Small on purpose: if JANET is far enough
# behind that four utterances are queued, the useful thing is to drop the oldest
# and stay current, not to work through a backlog answering questions from a
# minute ago.
MAX_QUEUED_UTTERANCES = 4

# JANET_DEBUG_AUDIO=1 saves every captured utterance as a WAV.
#
# Worth having because transcripts alone can't tell two very different problems
# apart: "the microphone only caught a fragment" (a VAD or level problem) and
# "Whisper mangled a perfectly good recording" (a model problem). The fix for
# each is in a different file, so guessing wastes time — listen to the audio and
# you know immediately.
DEBUG_AUDIO = os.environ.get("JANET_DEBUG_AUDIO", "").lower() in ("1", "true", "yes")
DEBUG_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "debug_audio"

# JANET_BARGE_IN=1 keeps the microphone live while JANET is speaking. Only
# useful with echo cancellation loaded, or JANET transcribes its own voice and
# answers itself — see the README.
BARGE_IN = os.environ.get("JANET_BARGE_IN", "").lower() in ("1", "true", "yes")


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


def _preload():
    """Load every model before the first utterance instead of during it.

    Measured: Whisper 2.0s + DistilBERT 1.6s + Silero 0.05s. Loaded lazily, that
    whole 3.6s landed on the first thing you said after starting JANET, which is
    exactly when it feels broken. Paid at startup it costs nothing — you aren't
    talking yet.
    """
    print("⏳ Warming up models...")
    SpeechDetector()                        # Silero, cached in a module singleton
    from audio.stt import _get_model
    _get_model()                            # Whisper
    from intent.classifier import predict
    predict("what time is it")              # DistilBERT
    print("✅ Ready.")


def _capture(utterances, stop_event):
    """Read the mic forever, emitting complete utterances onto `utterances`.

    Runs on its own thread and never does slow work, so the microphone is not
    at the mercy of how long Whisper, the language model or playback take.
    """
    detector = SpeechDetector()
    ring = RingBuffer(capacity=PRE_ROLL_SAMPLES)
    utterance = None                # None = idle; a list = actively collecting

    for frame in frames():
        if stop_event.is_set():
            return

        # JANET's own voice reaches the microphone, and without echo
        # cancellation it would be transcribed like anything else — JANET would
        # hear itself, answer itself, and do it again. So while it is speaking
        # (plus a short settle) we throw frames away and keep the VAD reset.
        if not BARGE_IN and speaker.is_speaking():
            if utterance is not None:
                utterance = None
                detector.reset()
            ring.clear()
            continue

        ring.push(frame)
        event = detector.process(frame)

        if utterance is None:
            if event == "start":
                # seed with the pre-roll so the opening word survives
                utterance = [ring.snapshot()]
            continue

        utterance.append(frame)
        if event == "end" or len(utterance) >= MAX_UTTERANCE_FRAMES:
            try:
                utterances.put_nowait(np.concatenate(utterance))
            except queue.Full:
                # Drop the OLDEST, keep the newest: the thing just said matters
                # more than something from a minute ago.
                try:
                    utterances.get_nowait()
                    utterances.put_nowait(np.concatenate(utterance))
                    print("⚠  running behind — dropped an older utterance")
                except (queue.Empty, queue.Full):
                    pass
            detector.reset()
            ring.clear()
            utterance = None


def _handle(audio, history):
    """Transcribe one collected utterance and respond to it."""
    query = transcribe(audio)
    print(f"🗣  You said: {query.strip()}")
    if DEBUG_AUDIO:
        _save_debug_audio(audio, query)
    if not query.strip():
        return                      # Whisper heard nothing intelligible; stay quiet
    ctx = Context(speak=speaker.speak, query=query, history=history)
    reply = respond(query, ctx)
    if reply is None:
        return                          # not addressed / not for JANET — stay silent
    print(f"⚙️  Reply: {reply.text}")
    if reply.reasoning:
        print(f"💭 Why: {reply.reasoning}")
    speaker.speak(reply.text)
    # Remember this addressed exchange so later questions have context — including
    # what the action returned and why JANET answered that way.
    history.add(query, reply.text, facts=reply.facts, reasoning=reply.reasoning)


def main():
    _preload()
    print("JANET is running (always-listening). Press Ctrl-C to quit.")
    if BARGE_IN:
        print("🎙  Barge-in ON — the mic stays live while JANET speaks.")

    history = ConversationHistory()     # short-term memory, shared across turns
    utterances = queue.Queue(maxsize=MAX_QUEUED_UTTERANCES)
    stop_event = threading.Event()

    speaker.start()

    # Bring back anything JANET promised for a future time. Without this a
    # restart silently threw away every alarm, timer and reminder — JANET said
    # the alarm was set, and it was, right up until it wasn't.
    restore_ctx = Context(speak=speaker.speak, query="", history=history)
    alarm_action.restore(restore_ctx)
    timer_action.restore(restore_ctx)
    reminder_action.restore(restore_ctx)
    listener = (threading.Thread(target=_capture, args=(utterances, stop_event),
                                name="janet-capture", daemon=True))
    listener.start()

    # The main thread does the thinking. Keeping it here (rather than on a third
    # worker) means Ctrl-C lands where it's easy to handle, and there is exactly
    # one thread touching the history and the confirmation gate — so neither
    # needs a lock.
    try:
        while True:
            try:
                audio = utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            _handle(audio, history)
    finally:
        stop_event.set()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 JANET stopped.")
        speaker.stop(timeout=2.0)
