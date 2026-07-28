# src/intent/eventparse.py
"""Parse a "put this on my calendar" request into a title + time + duration.

Pure and clock-free, like `timeparse`: it reports what was SAID. Turning the
spoken time into a real datetime is the action's job (it reuses
`alarm_action.resolve`, so "at 3" resolves to the soonest future 3 o'clock the
same way alarms do).
"""
import re

from intent import timeparse

# Saying any of these turns a calendar utterance from a question into a request
# to create something ("what's on my calendar" vs "add lunch to my calendar").
_CREATE_VERBS = ("schedule", "add", "create", "book", "set up", "put", "make")

# "for 30 minutes" / "for 2 hours"
_DURATION = re.compile(r"\bfor\s+(\d+)\s*(minutes?|mins?|hours?)\b")

# Phrases that describe WHEN, stripped out so only the title is left.
_TIME_PHRASES = [
    re.compile(r"\bat\s+\d{1,2}\s*[:.]?\s*\d{0,2}\s*(am|pm|a m|p m)?\b"),
    re.compile(r"\b(tomorrow|today|tonight)\b"),
    re.compile(r"\b(on|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    re.compile(r"\bfor\s+(\d+\s*(minutes?|mins?|hours?)|an?\s+hour|half\s+an\s+hour)\b"),
    re.compile(r"\b(oclock|o clock)\b"),
]

_CALENDAR_FILLER = re.compile(r"\b(to|on|in|into)?\s*(my|the)?\s*calendar\b")
# Words that carry no meaning in a title. "meeting"/"lunch"/"event" are KEPT —
# they make perfectly good titles ("schedule a meeting" -> "Meeting").
_FILLER_WORDS = re.compile(r"\b(a|an|the|my|please|new|janet)\b")

DEFAULT_DURATION_MINUTES = 60


def is_create(text):
    """True when the utterance asks to put something ON the calendar."""
    return any(verb in text.lower() for verb in _CREATE_VERBS)


def _duration_minutes(text):
    match = _DURATION.search(text)
    if match:
        amount = int(match.group(1))
        return amount * 60 if match.group(2).startswith("hour") else amount
    if re.search(r"\bfor\s+half\s+an\s+hour\b", text):
        return 30
    if re.search(r"\bfor\s+an\s+hour\b", text):
        return 60
    return DEFAULT_DURATION_MINUTES


def _title(text):
    """Whatever is left once the command, the time, and filler are removed."""
    for verb in _CREATE_VERBS:
        text = re.sub(rf"\b{verb}\b", " ", text)
    text = _CALENDAR_FILLER.sub(" ", text)
    for pattern in _TIME_PHRASES:
        text = pattern.sub(" ", text)
    text = _FILLER_WORDS.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)          # drop stray punctuation
    return " ".join(text.split())


def parse_event(text):
    """-> {"title": str, "time": <timeparse spec>|None, "duration_minutes": int}.

    `title` is "" and `time` is None when the utterance didn't say them; the
    handler asks rather than guessing.
    """
    text = text.lower()
    # timeparse works in digits, so give it the spoken hours it can't read.
    text = re.sub(r"\bnoon\b", "12 pm", text)
    text = re.sub(r"\bmidnight\b", "12 am", text)

    duration = _duration_minutes(text)

    spec = timeparse.parse_alarm(text)
    # parse_alarm reports "cancel" if the text contains a cancel word (a title
    # like "bus stop" would); only a real "set" with an hour counts as a time.
    time_spec = spec if spec.get("action") == "set" and spec.get("hour") is not None else None

    return {"title": _title(text), "time": time_spec, "duration_minutes": duration}
