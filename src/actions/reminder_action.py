# src/actions/reminder_action.py
"""Schedule spoken reminders — "remind me to call mom at 5 pm" / "in 10 minutes".

Same shape as alarm_action: pure phrasing helpers, then a small stateful
scheduler (a `Reminder` object + a module registry) and the handler. The clock
math for absolute times is NOT reimplemented here — `alarm_action.resolve`
already encodes the resolution rule (soonest-future meridiem inference, the
tomorrow floor, specific weekdays), so we import it.

> Reminders live in memory only. They are lost when JANET restarts, and there is
> no per-reminder cancel yet — "cancel my reminder" clears all of them.
"""
from datetime import datetime, timedelta
from threading import Timer

from actions.alarm_action import resolve, speak_time
from intent.remindparse import parse_reminder

_reminders = []          # active Reminder objects — the registry list + cancel need

# You remind someone TO do a thing, but ABOUT a thing. If the task starts like a
# noun phrase, "about" is the phrasing that doesn't read broken.
_NOUNISH = ("the ", "a ", "an ", "my ", "our ", "your ", "that ", "this ")

# Largest-first, so 600 seconds is said as "10 minutes" rather than "600 seconds".
_UNITS = (("week", 604800), ("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1))


def _remind_phrase(text):
    """'to call mom' / 'about the meeting' — the grammatical link to the task."""
    return f"about {text}" if text.startswith(_NOUNISH) else f"to {text}"


def _describe_delay(seconds):
    """'30 seconds' / '10 minutes' / '2 hours' — the biggest unit that divides evenly."""
    for unit, size in _UNITS:
        if seconds >= size and seconds % size == 0:
            count = seconds // size
            return f"{count} {unit}" if count == 1 else f"{count} {unit}s"
    return f"{seconds} seconds"


def _when_phrase(when):
    """'at 5 PM' / 'tomorrow at 9 AM' / 'on Friday at 8 AM', relative to today."""
    spoken = speak_time(when.hour, when.minute)
    today = datetime.now().date()
    if when.date() == today:
        return f"at {spoken}"
    if when.date() == today + timedelta(days=1):
        return f"tomorrow at {spoken}"
    return f"on {when:%A} at {spoken}"


class Reminder:
    """One pending reminder: what to say, when, and the timer that will say it."""

    def __init__(self, text, when):
        self.text = text
        self.when = when
        self.timer = None

    def arm(self, delay, ctx):
        # daemon so a pending reminder never blocks Ctrl-C / shutdown.
        self.timer = Timer(max(delay, 0), self.fire, args=(ctx,))
        self.timer.daemon = True
        self.timer.start()

    def fire(self, ctx):
        """Speak the reminder, then de-register — reminders are one-shot."""
        ctx.speak(f"Reminder: {self.text}.")
        if self in _reminders:
            _reminders.remove(self)


def _cancel_all():
    if not _reminders:
        return "You have no reminders to cancel."
    count = len(_reminders)
    for reminder in _reminders:
        if reminder.timer:
            reminder.timer.cancel()
    _reminders.clear()
    return "Cancelled the reminder." if count == 1 else f"Cancelled {count} reminders."


def _list_all():
    if not _reminders:
        return "You have no reminders."
    items = [f"{r.text} {_when_phrase(r.when)}" for r in _reminders]
    if len(items) == 1:
        return f"You have 1 reminder: {items[0]}."
    # "a, b, and c" — the comma before "and" gives the TTS a natural pause.
    listed = ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"You have {len(items)} reminders: {listed}."


def handle(slots, ctx):
    """REMINDER intent entry: set / list / cancel.

    Reads the RAW transcript from `ctx.query` (like calc_action and
    general_action) instead of a slot, so no slot filling is needed in
    intent._slots_for — the parser does its own normalization.
    """
    request = parse_reminder(ctx.query or "")
    if request["action"] == "cancel":
        return _cancel_all()
    if request["action"] == "list":
        return _list_all()

    text = request["text"]
    delay, spec = request["delay_seconds"], request["time"]
    if delay is None and spec is None:
        # Never guess a time — ask for whichever half is missing.
        if text:
            return f"When should I remind you {_remind_phrase(text)}?"
        return "What should I remind you about, and when?"
    if not text:
        return "What should I remind you about?"

    now = datetime.now()
    if delay is not None:
        when = now + timedelta(seconds=delay)
        confirmation = (
            f"Okay, I'll remind you {_remind_phrase(text)} in {_describe_delay(delay)}."
        )
    else:
        when, _ = resolve(now, spec)        # the alarm resolution rule, reused
        confirmation = f"Okay, I'll remind you {_remind_phrase(text)} {_when_phrase(when)}."

    reminder = Reminder(text, when)
    reminder.arm((when - now).total_seconds(), ctx)
    _reminders.append(reminder)
    return confirmation
