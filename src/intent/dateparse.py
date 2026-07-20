# src/intent/dateparse.py
"""Parse a calendar query into a date request (pure, no I/O)."""
from datetime import datetime, timedelta, time as dtime

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _day_range(day):
    start = datetime.combine(day, dtime(0, 0))
    end = start + timedelta(days=1) - timedelta(seconds=1)
    return start, end


def parse_query(text, now):
    """Return {"kind","start","end","label"}; kind in {list, next, free}."""
    text = text.lower()
    free = "free" in text or "am i busy" in text
    if not free and "next" in text and any(
        w in text for w in ("meeting", "event", "appointment", "calendar")
    ):
        return {"kind": "next", "start": now, "end": now + timedelta(days=30), "label": "next"}

    today = now.date()
    if "tomorrow" in text:
        start, end = _day_range(today + timedelta(days=1))
        label = "tomorrow"
    elif "week" in text:
        start = datetime.combine(today, dtime(0, 0))
        _, end = _day_range(today + timedelta(days=(6 - today.weekday())))   # through Sunday
        label = "this week"
    else:
        wd = next((idx for name, idx in _WEEKDAYS.items() if name in text), None)
        if wd is not None:
            day = today + timedelta(days=(wd - today.weekday()) % 7)
            start, end = _day_range(day)
            label = "on " + day.strftime("%A")
        else:
            start, end = _day_range(today)
            label = "today"
    return {"kind": "free" if free else "list", "start": start, "end": end, "label": label}
