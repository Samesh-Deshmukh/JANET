# src/actions/calendar_action.py
"""CALENDAR intent (slice 1: read-only). Reads the configured calendar and speaks
a summary. The backend is chosen by the factory (demo / CalDAV), so this file
never mentions a specific provider."""
from datetime import datetime, timedelta

import caldav.lib.error

from intent.dateparse import parse_query
from integrations.calendar_factory import get_calendar_source

# Narrow, operational failures only: server down / bad credentials. Real bugs
# still propagate (same posture as general_action's Ollama catch).
_CONN_ERRORS = (ConnectionError, caldav.lib.error.DAVError)


def _speak_time(dt):
    if dt.hour == 12 and dt.minute == 0:
        return "noon"
    h12 = dt.hour % 12 or 12
    ap = "AM" if dt.hour < 12 else "PM"
    return f"{h12}:{dt.minute:02d} {ap}" if dt.minute else f"{h12} {ap}"


def _speak_when(event, now):
    """'today at 9 AM' / 'tomorrow at 3 PM' / 'Friday at 10 AM'."""
    day = event.start.date()
    if day == now.date():
        d = "today"
    elif day == now.date() + timedelta(days=1):
        d = "tomorrow"
    else:
        d = event.start.strftime("%A")
    return f"{d}, all day" if event.all_day else f"{d} at {_speak_time(event.start)}"


def _item(e):
    return f"{e.summary} (all day)" if e.all_day else f"{e.summary} at {_speak_time(e.start)}"


def _join(parts):
    """Natural spoken list: 'A', 'A and B', 'A, B, and C'."""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _format_list(events, label):
    if not events:
        return f"Nothing on your calendar {label}."
    n = len(events)
    shown = [_item(e) for e in events[:5]]
    # Long days get capped so JANET doesn't read out twenty events.
    body = ", ".join(shown) + f", and {n - 5} more" if n > 5 else _join(shown)
    plural = "s" if n != 1 else ""
    return f"You have {n} event{plural} {label}: {body}."


def _format_next(events, now):
    upcoming = [e for e in events if e.start >= now] or events
    if not upcoming:
        return "You have nothing coming up."
    e = upcoming[0]
    return f"Your next event is {e.summary}, {_speak_when(e, now)}."


def _format_free(events, label):
    if not events:
        return f"You're free {label}."
    n = len(events)
    return f"You have {n} event{'s' if n != 1 else ''} {label}."


def handle(slots, ctx):
    source = get_calendar_source()
    if source is None:
        return "Your calendar isn't set up yet."
    now = datetime.now()
    req = parse_query(ctx.query.lower(), now)
    try:
        events = source.events_between(req["start"], req["end"])
    except _CONN_ERRORS:
        return "I couldn't reach your calendar."
    if req["kind"] == "next":
        return _format_next(events, now)
    if req["kind"] == "free":
        return _format_free(events, req["label"])
    return _format_list(events, req["label"])
