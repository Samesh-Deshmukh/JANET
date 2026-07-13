"""The addressing gate: was this utterance aimed at JANET, or ambient speech?

This is the FIRST of JANET's two veto layers (the classifier's NONE class is the
second). It scores an utterance from linguistic signals and compares the total to
a threshold. Weights follow the spec (JANET_v2.0.md 4.2); the threshold is tuned
down from the spec's 50 to 40 because we can't yet compute its environmental
signals (person-count +40, phone, TV) — those need sensors that don't exist yet.

`score()` is a pure function of the text so it's easy to test and tune. The
respond() pipeline prints the breakdown so decisions are visible on real hardware.

Deliberately NOT here yet (deferred): conversation-continuation (+50 if JANET
spoke recently) and every sensor-based signal.
"""
import re

THRESHOLD = 40  # total >= THRESHOLD -> treat as addressed to JANET

# First word signals speech shape. Kept as sets for O(1) "does it start with…".
_QUESTION_WORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "is", "are", "am", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "has", "have",
}
_COMMAND_WORDS = {
    "turn", "set", "play", "remind", "tell", "start", "stop", "cancel", "add",
    "open", "close", "lock", "unlock", "wake", "mute", "dim", "switch", "show",
    "give", "send", "read", "check", "schedule", "pause", "resume", "delete",
    "remove", "snooze", "create", "calculate",
}
# Words tied to things JANET actually does — weak evidence it's a real request.
_TASK_KEYWORDS = {
    "time", "date", "day", "timer", "alarm", "remind", "reminder", "weather",
    "forecast", "temperature", "light", "lights", "thermostat", "email",
    "calendar", "meeting", "appointment", "calculate", "news",
}


def score(query):
    """Return (total, breakdown) where breakdown lists the (signal, points) that fired."""
    q = query.lower().strip()
    words = q.split()
    first = words[0] if words else ""
    tokens = set(words)
    breakdown = []

    if "janet" in q or "janice" in q:                 # direct address (or common mishear)
        breakdown.append(("janet", 50))
    if re.search(r"\b(am|do|have|can)\s+i\b", q) or "my" in tokens or "me" in tokens:
        breakdown.append(("self-ref", 35))            # about the user's own stuff
    if first in _QUESTION_WORDS:
        breakdown.append(("question", 25))
    if first in _COMMAND_WORDS:
        breakdown.append(("command", 25))
    if tokens & _TASK_KEYWORDS:
        breakdown.append(("keyword", 15))
    if tokens & {"you", "your", "yourself"}:
        breakdown.append(("2nd-person", 10))

    total = sum(points for _, points in breakdown)
    return total, breakdown


def is_addressed(query):
    """True if the utterance scores at or above the threshold."""
    total, _ = score(query)
    return total >= THRESHOLD
