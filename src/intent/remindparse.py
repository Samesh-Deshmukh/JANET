# src/intent/remindparse.py
"""Turn a REMINDER utterance into a structured request (pure, no clock, no I/O).

A reminder has two halves — WHAT to be reminded of and WHEN — and the words for
each are tangled together ("remind me to call mom at 5 pm"). This file's whole
job is to pull them apart:

    {"action": "cancel"}                     cancel/clear my reminder(s)
    {"action": "list"}                       what are my reminders
    {"action": "set", "text": "call mom",    everything else
     "delay_seconds": None, "time": {...}}

`delay_seconds` covers relative times ("in 10 minutes"); `time` holds an
absolute spec produced by `timeparse.parse_alarm` (the same dict the alarm
handler consumes). At most one of the two is filled; both are `None` when no
time was spoken, and the handler then asks.

Like `timeparse`, this file only *understands the words* — it never reads the
clock or arms a timer. That belongs to `actions/reminder_action.py`.
"""
import re

from intent.normalize import normalize
from intent.numwords import words_to_numbers
from intent.timeparse import parse_alarm

# ---------------------------------------------------------------------------
# Which of the three requests is this?
# ---------------------------------------------------------------------------

# A cancel verb followed (within a few words) by the word "reminder". Requiring
# the noun is what keeps "remind me to cancel the dentist appointment" a SET:
# there is no "reminder" after "cancel" there, so this doesn't match.
_CANCEL_RE = re.compile(
    r"\b(?:cancel|stop|clear|delete|remove)\b(?:\s+\w+){0,3}?\s+reminders?\b"
)

# A question/listing opener. Anchored at the START so "set a reminder to show
# my slides" (which contains "show") is not mistaken for a listing request.
_LIST_RE = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?(?:janet\s+)?"
    r"(?:what|which|how\s+many|list|show|tell|do\s+i\s+have|are\s+there)\b"
)
_REMINDER_NOUN_RE = re.compile(r"\breminders?\b")

# ---------------------------------------------------------------------------
# Peeling the command words off the front
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"^(?:hey|ok|okay|yo|janet|please)\b\s*"
    r"|^(?:can|could|would|will)\s+you\b\s*"
)
_COMMAND_RE = re.compile(
    r"^(?:"
    r"remind\s+me"
    r"|(?:set|add|create|make|put)\s+(?:up\s+)?(?:a|an|the)?\s*reminder"
    r"|(?:dont|do\s+not)\s+(?:let\s+me\s+)?forget"
    r")\b\s*"
)
_CONNECTOR_RE = re.compile(r"^(?:to|about|that|for|of)\b\s*")

# Dangling words that can be left at either end once the time phrase is cut out
# ("call mom at" -> "call mom").
_LEAD_EDGE = ("to", "about", "that", "for", "of", "and")
_TAIL_EDGE = ("at", "on", "in", "by", "around", "to", "for", "about", "and",
              "this", "next", "every")

# ---------------------------------------------------------------------------
# Time phrases
# ---------------------------------------------------------------------------

_UNIT_SECONDS = {
    "second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800,
}

# "in 10 minutes" / "in an hour" / "in half an hour". Spoken numbers were already
# turned into digits by words_to_numbers, so only digits and a/an/half remain.
_RELATIVE_RE = re.compile(
    r"\bin\s+(\d+|half\s+an?|half|an?)\s+(second|minute|hour|day|week)s?\b"
)

_DAY = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# Every phrase that means "a point in time". Each is deliberately narrow — a
# preposition alone is never enough ("at" only counts when a number follows it),
# because anything matched here gets CUT OUT of the reminder text.
_TIME_PATTERNS = [
    # "at 5", "at 5 30", "by 7 am", "around 10 oclock", compact "at 505"
    r"\b(?:at|around|by)\s+\d{1,4}(?:\s+\d{2})?\s*(?:[ap]\s*m\b|oclock\b)?",
    # a bare clock time that names am/pm or o'clock ("3 pm", "10 oclock")
    r"\b\d{1,4}\s*(?:[ap]\s*m\b|oclock\b)",
    # "noon" / "midnight" — translated to a clock time before parsing
    r"\b(?:at\s+)?(?:noon|midday|midnight)\b",
    # a day word, optionally with the part of day ("tomorrow morning")
    r"\b(?:tomorrow|today|tonight)(?:\s+(?:morning|afternoon|evening|night))?\b",
    # recurrence: "every day", "every weekday", "every monday"
    r"\b(?:every|each)\s+(?:day|morning|afternoon|evening|night|weekday|weekend|"
    + _DAY + r")s?\b",
    # "on monday", "next friday", "this weekend"
    r"\b(?:on|next|this)\s+(?:weekday|weekend|week|" + _DAY + r")s?\b",
    # a bare weekday ("remind me monday to call the bank")
    r"\b(?:" + _DAY + r")s?\b",
]
_TIME_RE = re.compile("|".join(_TIME_PATTERNS))

# parse_alarm only understands digits, so give it some.
_WORD_TIMES = {"noon": "12 pm", "midday": "12 pm", "midnight": "12 am"}
_WORD_TIME_RE = re.compile(r"\b(?:noon|midday|midnight)\b")


def _strip_command(text):
    """Peel the leading command words off until only the task is left.

    Loops because the phrasings nest: "dont forget to remind me to feed the cat"
    needs two rounds. Bounded so a weird sentence can never spin forever.
    """
    for _ in range(4):
        stripped = _FILLER_RE.sub("", text, count=1).strip()
        stripped = _COMMAND_RE.sub("", stripped, count=1).strip()
        stripped = _CONNECTOR_RE.sub("", stripped, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return text


def _amount_value(word):
    """'10' -> 10, 'a'/'an' -> 1, 'half an' -> 0.5."""
    word = word.strip()
    if word.startswith("half"):
        return 0.5
    if word in ("a", "an"):
        return 1
    return int(word)


def _parse_relative(text):
    """(delay_seconds, leftover_text) for an "in <n> <unit>" phrase, else (None, text)."""
    match = _RELATIVE_RE.search(text)
    if not match:
        return None, text
    seconds = int(_amount_value(match.group(1)) * _UNIT_SECONDS[match.group(2)])
    leftover = text[:match.start()] + " " + text[match.end():]
    return seconds, leftover


def _split_time(text):
    """(task_text, time_text): cut EVERY time phrase out of `text`.

    Cutting all of them (rather than stopping at the first) is what makes
    "tomorrow at 9 am" work — two separate phrases that together describe one
    moment. What's removed becomes the time text; what's left is the task.
    """
    found = []

    def _cut(match):
        found.append(match.group(0).strip())
        return " "                      # a space, so the words either side stay apart

    task = _TIME_RE.sub(_cut, text)
    return task, " ".join(found)


def _parse_absolute(time_text):
    """Reuse the alarm parser on the extracted time words; None if it found no hour.

    Only the time phrase is passed in — never the whole sentence. parse_alarm
    reads "stop"/"cancel" anywhere in its input as a cancel request, so a task
    like "stop the dishwasher" would otherwise hijack it.
    """
    time_text = _WORD_TIME_RE.sub(lambda m: _WORD_TIMES[m.group(0)], time_text)
    spec = parse_alarm(time_text)
    if spec["action"] != "set" or spec["hour"] is None:
        return None
    return spec


def _clean(text):
    """Collapse whitespace and drop connector words dangling at either end."""
    words = text.split()
    while words and words[0] in _LEAD_EDGE:
        words.pop(0)
    while words and words[-1] in _TAIL_EDGE:
        words.pop()
    return " ".join(words)


def parse_reminder(text):
    """Parse a REMINDER utterance into a request dict (see the module docstring)."""
    # The handler passes the RAW transcript, so canonicalize here: normalize()
    # gives the lowercase, punctuation-free form the patterns below assume, and
    # words_to_numbers turns "in fifteen minutes"/"at five" into digits so one
    # set of numeric patterns covers spoken and written numbers alike.
    text = words_to_numbers(normalize(text))

    if _CANCEL_RE.search(text):
        return {"action": "cancel"}
    if _LIST_RE.search(text) and _REMINDER_NOUN_RE.search(text):
        return {"action": "list"}

    body = _strip_command(text)
    delay, body = _parse_relative(body)
    body, time_text = _split_time(body)
    # A relative delay wins: "in 10 minutes" is already a complete answer, and
    # feeding "10" to the absolute parser would read it as an hour.
    spec = _parse_absolute(time_text) if delay is None and time_text else None
    return {
        "action": "set",
        "text": _clean(body),
        "delay_seconds": delay,
        "time": spec,
    }
