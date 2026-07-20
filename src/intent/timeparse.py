# src/intent/timeparse.py
"""Turn a normalized transcript into a structured alarm spec (pure, no clock).

Runs on the SAME normalized text the classifier sees, so the colon is already
gone ("7:30 am" -> "7 30 am"). Splitting this linguistic parse out of the
scheduler keeps both halves small and unit-testable.
"""
import re

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
ALL_DAYS = frozenset(range(7))
WEEKDAY_SET = frozenset({0, 1, 2, 3, 4})
WEEKEND_SET = frozenset({5, 6})

_CANCEL_WORDS = ("cancel", "stop", "delete", "remove", "turn off")


def _parse_time(text):
    """(hour, minute, meridiem) from the normalized time forms, or (None, 0, None).
    hour is as spoken (1-12), or 24-hour when a >=13 form is used."""
    # Whisper writes "p.m."/"a.m.", which normalize turns into spaced "p m"/"a m";
    # collapse those back so meridiem matching is uniform ("505 p m" -> "505 pm").
    text = re.sub(r"\b([ap])\s*m\b", r"\1m", text)
    m = re.search(r"\b(\d{1,2})\s+(\d{2})\s*(am|pm)\b", text)   # "7 30 am"
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    # Compact "HHMM"/"HMM" — Whisper drops the colon ("5:05" -> "505", "12:30" -> "1230").
    m = re.search(r"\b(\d{3,4})\s*(am|pm)?\b", text)
    if m:
        digits = m.group(1)
        return int(digits[:-2]), int(digits[-2:]), m.group(2)
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text)             # "7 am" / glued "7am"
    if m:
        return int(m.group(1)), 0, m.group(2)
    m = re.search(r"\b(\d{1,2})\s*oclock\b", text)              # "10 oclock"
    if m:
        return int(m.group(1)), 0, None
    m = re.search(r"\b(\d{1,2})\s+(\d{2})\b", text)             # "6 45" / "18 30"
    if m:
        return int(m.group(1)), int(m.group(2)), None
    m = re.search(r"\b(\d{1,2})\b", text)                       # bare hour "7"
    if m:
        return int(m.group(1)), 0, None
    return None, 0, None


def _parse_recurrence(text):
    """(weekdays, recurring, tomorrow). Default: all days, one-shot, not tomorrow."""
    if "every day" in text or "everyday" in text or "daily" in text:
        return ALL_DAYS, True, False
    if "weekday" in text:
        return WEEKDAY_SET, True, False
    if "weekend" in text:
        return WEEKEND_SET, True, False
    for name, idx in _WEEKDAYS.items():
        if name in text:
            # "every monday" or plural "mondays" -> recurring; "on monday" -> one-shot
            recurring = (f"every {name}" in text) or (f"{name}s" in text)
            return frozenset({idx}), recurring, False
    if "tomorrow" in text:
        return ALL_DAYS, False, True
    return ALL_DAYS, False, False


def parse_alarm(text):
    """Parse a normalized ALARM utterance into a spec dict (see module docstring)."""
    text = text.lower()
    if any(w in text for w in _CANCEL_WORDS):
        return {"action": "cancel"}
    hour, minute, meridiem = _parse_time(text)
    if hour is not None:
        valid = (1 <= hour <= 12) if meridiem else (0 <= hour <= 23)
        if not valid or not (0 <= minute <= 59):
            hour = None
    weekdays, recurring, tomorrow = _parse_recurrence(text)
    return {
        "action": "set",
        "hour": hour,
        "minute": minute,
        "meridiem": meridiem,
        "weekdays": weekdays,
        "recurring": recurring,
        "tomorrow": tomorrow,
    }
