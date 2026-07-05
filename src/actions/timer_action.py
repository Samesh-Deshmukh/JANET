# src/actions/timer_action.py
from threading import Timer


def handle(slots, ctx):
    seconds = slots.get("duration", 300)
    # Fire deferred speech when the timer elapses — action stays non-blocking.
    Timer(seconds, lambda: ctx.speak("Timer finished!")).start()
    return f"Timer set for {seconds} seconds."
