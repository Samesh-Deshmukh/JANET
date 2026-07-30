# src/audio/speaker.py
"""Speech output on its own thread, so nothing waits for JANET's mouth.

Measured on this machine, a typical calendar answer takes **8.5 seconds** to
play — against 0.10s for Whisper and ~2s for the language model. Speaking is by
far the longest thing JANET does, and until now the whole program sat inside it:
the microphone was deaf for the entire 8.5s.

This module owns one worker thread and a queue. `speak()` returns immediately.

**Everything that talks goes through here**, which fixes a second bug for free.
Alarms, timers and reminders fire on their own `threading.Timer` threads and
called `ctx.speak` directly, so two of them landing together — or one landing
while JANET was answering you — played *simultaneously*, over the top of each
other. A single consumer thread serialises them into a queue instead, and none
of those files needed changing: they were already calling `ctx.speak`.

`is_speaking()` is what keeps JANET from hearing itself. There is no echo
cancellation by default, so the capture thread throws frames away while this is
true (see `main.py`).
"""
import queue
import threading
import time

from audio import tts

# Enough to hold a reply plus a couple of alarms going off together. Bounded on
# purpose: if speech is backing up, the useful thing is to notice and drop it,
# not to build a minute-long monologue nobody will wait for.
MAX_QUEUED = 8

# Speakers reach the microphone a moment after playback ends — the tail of a
# word, then the room. Staying deaf for a beat afterwards stops that tail
# starting a fresh utterance the instant JANET stops talking.
SETTLE_S = 0.3

_queue = queue.Queue(maxsize=MAX_QUEUED)
_thread = None
_lock = threading.Lock()
_pending = 0                 # queued + currently playing
_quiet_until = 0.0           # monotonic deadline for the settle window
_STOP = object()             # sentinel that ends the worker


def _worker():
    global _pending, _quiet_until
    while True:
        item = _queue.get()
        if item is _STOP:
            _queue.task_done()
            return
        try:
            tts.say(item)
        except Exception as exc:            # never let one bad phrase kill audio
            print(f"⚠  couldn't speak: {exc}")
        finally:
            with _lock:
                _pending -= 1
                _quiet_until = time.monotonic() + SETTLE_S
            _queue.task_done()


def start():
    """Start the speech thread if it isn't already running. Safe to call twice."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_worker, name="janet-speech", daemon=True)
        _thread.start()


def speak(phrase):
    """Queue `phrase` to be spoken. Returns immediately.

    This is the `speak` handed to every handler as `Context.speak`, so a timer
    firing mid-sentence queues behind the sentence instead of talking over it.
    """
    if not phrase:
        return
    start()
    global _pending
    # Counted BEFORE the put: the worker can pick the item up and decrement the
    # moment it lands, and incrementing afterwards would let the count go
    # negative and make is_speaking() briefly lie.
    with _lock:
        _pending += 1
    try:
        _queue.put_nowait(phrase)
    except queue.Full:
        with _lock:
            _pending -= 1
        print(f"⚠  speech queue full — dropped: {phrase[:60]!r}")


def is_speaking():
    """True while anything is queued, playing, or still settling.

    The capture thread uses this to discard microphone frames, which is the only
    thing stopping JANET transcribing its own voice and answering itself.
    """
    with _lock:
        return _pending > 0 or time.monotonic() < _quiet_until


def wait_until_quiet(timeout=10.0):
    """Block until the queue has drained. Used on shutdown so a goodbye is heard."""
    deadline = time.monotonic() + timeout
    while is_speaking() and time.monotonic() < deadline:
        time.sleep(0.05)


def stop(timeout=10.0):
    """Finish what's queued, then end the worker thread."""
    global _thread
    if _thread is None or not _thread.is_alive():
        return
    wait_until_quiet(timeout)
    _queue.put(_STOP)
    _thread.join(timeout=2.0)
    _thread = None
