"""The addressing gate: was this utterance aimed at JANET, or ambient speech?

This is the FIRST of JANET's two veto layers (the classifier's NONE class is the
second). It scores an utterance from linguistic signals and compares the total to
a threshold. Weights are informed by the spec (JANET_v2.0.md 4.2) but tuned for
what we can actually compute today.

Assumes NORMALIZED input (see normalize.py) — lowercase, no punctuation,
apostrophes removed. `respond()` normalizes before calling us, so "What's the
time?" arrives as "whats the time" and the word matching below actually fires.

Because a wrong "yes" here is cheap (the classifier's NONE class is a second
veto), a clear question or command scores enough to pass on its own; the
classifier then rejects non-JANET questions ("can you pass the salt"). What the
scorer uniquely adds: the "janet" address signal (the classifier never learned
it) and rejecting intent-like STATEMENTS ("i set my alarm wrong yesterday").

Deferred (need sensors / dialogue state): conversation-continuation (+50 if JANET
spoke recently), person-count, phone, TV/music exclusion.
"""
import re

THRESHOLD = 40  # total >= THRESHOLD -> treat as addressed to JANET

# First meaningful word signals speech shape. Contraction forms included because
# Whisper writes "what's" -> normalized "whats", not "what".
_QUESTION_WORDS = {
    "what", "whats", "when", "whens", "where", "wheres", "who", "whos", "whom",
    "which", "why", "whys", "how", "hows",
    "is", "are", "am", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "has", "have",
}
_COMMAND_WORDS = {
    "turn", "set", "play", "remind", "tell", "start", "stop", "cancel", "add",
    "open", "close", "lock", "unlock", "wake", "mute", "dim", "switch", "show",
    "give", "send", "read", "check", "schedule", "pause", "resume", "delete",
    "remove", "snooze", "create", "calculate",
}
# Words tied to things JANET actually does — evidence it's a real request for it.
_TASK_KEYWORDS = {
    "time", "date", "day", "timer", "alarm", "remind", "reminder", "weather",
    "forecast", "temperature", "light", "lights", "thermostat", "email",
    "calendar", "meeting", "appointment", "calculate", "news",
}
# Skipped when finding the "head" word, so "hey janet what time…" still reads as a
# question. (The +50 janet points come from the substring check, not the head.)
_LEAD_FILLER = {"hey", "ok", "okay", "so", "um", "uh", "well", "janet", "janice", "please", "yo"}


def _head_word(words):
    """First word after any leading filler ('' if none)."""
    i = 0
    while i < len(words) and words[i] in _LEAD_FILLER:
        i += 1
    return words[i] if i < len(words) else ""


def score(query):
    """Return (total, breakdown) where breakdown lists the (signal, points) that fired.

    Expects normalized text; lowercases/splits defensively in case a caller forgets.
    """
    q = query.lower().strip()
    words = q.split()
    tokens = set(words)
    head = _head_word(words)
    breakdown = []

    if "janet" in q or "janice" in q:                     # direct address (or common mishear)
        breakdown.append(("janet", 50))
    if re.search(r"\b(am|do|have|can|will)\s+i\b", q) or "my" in tokens or "me" in tokens:
        breakdown.append(("self-ref", 30))                # about the user's own stuff
    if head in _QUESTION_WORDS:
        breakdown.append(("question", 40))
    if head in _COMMAND_WORDS:
        breakdown.append(("command", 40))
    if tokens & _TASK_KEYWORDS:
        breakdown.append(("keyword", 25))
    if tokens & {"you", "your", "yourself"}:
        breakdown.append(("2nd-person", 10))

    total = sum(points for _, points in breakdown)
    return total, breakdown


def is_addressed(query):
    """True if the utterance scores at or above the threshold."""
    total, _ = score(query)
    return total >= THRESHOLD
