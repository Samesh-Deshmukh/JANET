# src/intent/timerparse.py
"""Turn a TIMER utterance into a structured request (pure — no clock, no threads).

Four things get asked of a timer, and telling them apart is purely a words job:

    {"action": "set",         "seconds": 300, "label": "pasta"}   start one
    {"action": "query",       "seconds": None, "label": None}     how much is left
    {"action": "cancel",      "seconds": None, "label": None}     cancel / stop it
    {"action": "unsupported", "seconds": None, "label": None}     pause / extend / reset

`seconds` is None when no duration was spoken — the handler then falls back to
the slot `intent._slots_for` filled. `label` is the optional name of the timer
("pasta timer", "timer for the eggs") so a timer can say which one it was.

Like `timeparse`/`remindparse`, this file only *understands the words*. It never
reads the clock or arms a countdown — that belongs to `actions/timer_action.py`.
"""
import re

from intent.normalize import normalize
from intent.numwords import words_to_numbers

# ---------------------------------------------------------------------------
# Which of the four requests is this?
# ---------------------------------------------------------------------------

# A cancel verb followed (within a few words) by the noun "timer". Requiring the
# noun is what keeps "set a timer to stop the wash cycle" a SET: there is no
# "timer" after "stop" there, so this doesn't match.
_CANCEL_RE = re.compile(
    r"\b(?:cancel|stop|clear|delete|remove|kill|end)\b(?:\s+\w+){0,3}?\s+timers?\b"
)

# Things you could reasonably ask of a timer that JANET can't do yet. Worth
# matching explicitly: without this, "pause the timer" falls through to SET and
# silently starts a brand-new 5-minute timer — much worse than saying "I can't".
_UNSUPPORTED_RE = re.compile(
    r"\b(?:pause|resume|restart|reset|snooze)\b(?:\s+\w+){0,3}?\s+timers?\b"
    r"|\b(?:add|subtract|extend|shorten|lengthen)\b.*\btimers?\b"
    r"|\bmake\s+the\s+timers?\s+(?:longer|shorter)\b"
)

# Openers that make the sentence a question about a timer rather than a command
# to start one. Anchored at the START so "set a timer to check the oven" (which
# contains "check") is not mistaken for a status question.
_QUESTION_RE = re.compile(
    r"^(?:how|what|whats|which|hows|is|are|do\s+i|did|got|list|show|tell|check)\b"
)
_TIMER_NOUN_RE = re.compile(r"\btimers?\b")
# "how much time is left" never says the word "timer" — these words carry it.
_STATUS_RE = re.compile(r"\b(?:left|remaining|running|going|status|done|up|longer)\b")

# Politeness/address words that sit in front of the real opener.
_FILLER_RE = re.compile(
    r"^(?:hey|ok|okay|yo|janet|please)\b\s*"
    r"|^(?:can|could|would|will)\s+you\b\s*"
    # Conversational run-ups. Found in real use: "actually how long is left on
    # that" wasn't recognised as a question, fell through to SET, and silently
    # started an unwanted 5-minute timer. Anything in front of the real opener
    # has to come off, or a mis-parse quietly *creates* a timer.
    r"|^(?:actually|so|and|but|um|uh|well|wait|hang\s+on|alright|right)\b\s*"
)

# ---------------------------------------------------------------------------
# How long?
# ---------------------------------------------------------------------------

_UNIT_SECONDS = {
    "hour": 3600, "hr": 3600, "minute": 60, "min": 60, "second": 1, "sec": 1,
}

# "<amount> <unit>" — "5 minutes", "a minute", "an hour", "half an hour". Spoken
# numbers are already digits by the time this runs (words_to_numbers), so only
# digits and the a/an/half wordings are left to spell out. "half" comes before
# "an?" in the alternation so "half an hour" isn't read as just "an hour".
_DURATION_RE = re.compile(
    r"\b(\d+|half\s+an?|half|an?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b"
)

# "set a SECOND timer for five minutes" means ANOTHER timer — "second" is an
# ordinal there, not a unit. Cut that exact phrase out before measuring, or the
# duration comes back as 301 seconds (1 + 300).
_ORDINAL_SECOND_RE = re.compile(r"\ban?\s+second\s+timers?\b")

# ---------------------------------------------------------------------------
# What is it called?
# ---------------------------------------------------------------------------

# "<name> timer" — the word directly in front of the noun ("pasta timer").
_LABEL_BEFORE_RE = re.compile(r"\b([a-z]+)\s+timers?\b")
# "timer for the <name>" — one or two words after the article ("for the eggs",
# "for the wash cycle"). Digits can't match [a-z], so the duration that usually
# follows ("...for the eggs 6 minutes") is naturally excluded.
_LABEL_AFTER_RE = re.compile(
    r"\btimers?\s+for\s+(?:the|my)\s+([a-z]+(?:\s+[a-z]+)?)\b"
)

# Words that can sit next to "timer" without naming it: units, articles,
# adjectives, another word for "timer", and the address words ("hey janet timer
# for two minutes" must not create a timer called "janet").
_NOT_A_LABEL = set(_UNIT_SECONDS) | {
    "hours", "hrs", "minutes", "mins", "seconds", "secs",
    "a", "an", "the", "my", "another", "other", "new", "quick", "long",
    "and", "for", "set", "start", "that", "this", "countdown",
    "hey", "ok", "okay", "yo", "janet", "please",
}


def _strip_filler(text):
    """Peel "hey janet" / "can you" off the front so the opener is first.

    Loops because they stack ("hey janet can you ..."); bounded so a weird
    sentence can never spin forever.
    """
    for _ in range(3):
        stripped = _FILLER_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return text


def _is_query(text):
    """True for "how much time is left" / "whats my timer at" / "check the timer"."""
    text = _strip_filler(text)
    if not _QUESTION_RE.search(text):
        return False
    # A question still has to be ABOUT a timer: either it says the word, or it
    # uses one of the status words that stand in for it.
    return bool(_TIMER_NOUN_RE.search(text) or _STATUS_RE.search(text))


def _amount(word):
    """'30' -> 30, 'a'/'an' -> 1, 'half an' -> 0.5."""
    word = word.strip()
    if word.startswith("half"):
        return 0.5
    if word in ("a", "an"):
        return 1
    return int(word)


def _parse_duration(text):
    """Total seconds of every "<amount> <unit>" in the sentence, or None.

    Summing every match (rather than stopping at the first) is what makes
    "1 minute 30 seconds" and "one hour and thirty minutes" work — two phrases
    that together describe one length.
    """
    total = 0
    text = _ORDINAL_SECOND_RE.sub(" ", text)
    for amount, unit in _DURATION_RE.findall(text):
        total += _amount(amount) * _UNIT_SECONDS[unit.rstrip("s")]
    return int(total) or None       # 0 means nothing was found -> None


def _parse_label(text):
    """The timer's name ("pasta", "wash cycle"), or None if it wasn't named."""
    for match in _LABEL_BEFORE_RE.finditer(text):
        word = match.group(1)
        if word not in _NOT_A_LABEL:
            return word
    match = _LABEL_AFTER_RE.search(text)
    if match:
        words = [w for w in match.group(1).split() if w not in _NOT_A_LABEL]
        return " ".join(words) or None
    return None


def _request(action, seconds=None, label=None):
    """Every branch returns the same shape, so the handler can read it blindly."""
    return {"action": action, "seconds": seconds, "label": label}


def parse_timer(text):
    """Parse a TIMER utterance into a request dict (see the module docstring)."""
    # The handler passes the RAW transcript, so canonicalize here: normalize()
    # gives the lowercase, punctuation-free form the patterns below assume, and
    # words_to_numbers turns "five minutes" into "5 minutes" so one set of
    # numeric patterns covers spoken and written numbers alike.
    text = words_to_numbers(normalize(text))

    if _CANCEL_RE.search(text):
        return _request("cancel")
    if _UNSUPPORTED_RE.search(text):
        return _request("unsupported")
    if _is_query(text):
        return _request("query")
    return _request("set", seconds=_parse_duration(text), label=_parse_label(text))
