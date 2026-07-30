# src/utils/confirm.py
"""A one-step spoken confirmation gate for actions that are hard to undo.

Voice input is unreliable — Whisper mishears constantly in this project ("set an
alarm" has come out as "Setenolarm"). So anything that writes to the outside
world states what it is about to do and only does it if the VERY NEXT thing you
say is a yes.

Deliberately simple and predictable (the owner must be able to explain it):
- one pending action at a time, single-shot (any answer consumes it),
- only an EXACT short phrase counts as yes/no — "yes", "do it", "cancel". A
  longer sentence is treated as a new command instead, so saying "stop the
  alarm" while a confirmation is pending doesn't get swallowed as "no",
- it expires (TIMEOUT_SECONDS) so a stray "yes" minutes later can't fire it.
"""
import re
import time

# Exact normalized phrases only — no fuzzy matching, no partial words.
_YES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed",
    "do it", "go ahead", "yes please", "please do", "correct", "thats right",
    "yes do it", "sounds good", "that works", "perfect",
}
_NO = {
    "no", "nope", "cancel", "cancel that", "never mind", "nevermind",
    "forget it", "no thanks", "dont", "do not", "stop that",
    # People decline by reassuring you, not by refusing. Live testing produced
    # "Oh no, it's okay." to a confirmation, which matched nothing and was
    # discarded as ambient speech — the question just went unanswered.
    "no its okay", "no thats okay", "no im okay", "its okay", "thats okay",
    "no its fine", "thats fine", "leave it",
}

# Speech doesn't start cleanly. "Oh yeah", "um, yes", "well, go ahead" all mean
# yes, and an exact-match set sees none of them. Stripped repeatedly so "oh, um,
# yeah" also lands on "yeah". Deliberately only fillers with no meaning of their
# own — nothing here can flip an answer.
_LEADING_FILLER = re.compile(r"^(?:oh|ah|um|uh|er|well|so|hmm|right|okay so|yeah so)\s+")
_TRAILING_FILLER = re.compile(r"\s+(?:please|thanks|thank you|then|mate)$")


def _tidy(text):
    """Strip conversational padding so an exact match has a chance."""
    text = (text or "").strip()
    previous = None
    while text != previous:
        previous = text
        text = _LEADING_FILLER.sub("", text)
        text = _TRAILING_FILLER.sub("", text)
    return text.strip()

TIMEOUT_SECONDS = 60

_pending = None          # {"run": callable, "created": float} or None


def request(run):
    """Remember an action to run if the next utterance confirms it.

    `run` is a zero-argument callable returning the sentence to speak once the
    action has actually been performed.
    """
    global _pending
    _pending = {"run": run, "created": time.time()}


def is_pending():
    """True when a confirmation is still awaited. Expired requests are dropped."""
    global _pending
    if _pending is None:
        return False
    if time.time() - _pending["created"] > TIMEOUT_SECONDS:
        _pending = None
        return False
    return True


def clear():
    """Forget any pending action (used by tests and by an explicit reset)."""
    global _pending
    _pending = None


def resolve(text):
    """Interpret `text` as the answer to the pending question.

    Returns the sentence to speak, or None when `text` wasn't a yes/no at all —
    in which case the caller should handle it as a normal command. Either way the
    pending action is consumed, so it can never fire later by accident.
    """
    global _pending
    if not is_pending():
        return None
    pending = _pending
    _pending = None                      # single-shot, whatever the answer was
    answer = _tidy(text)                 # "oh yeah" -> "yeah"
    if answer in _YES:
        return pending["run"]()
    if answer in _NO:
        return "Okay, cancelled."
    return None                          # not an answer -> treat as a new command
