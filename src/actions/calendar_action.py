# src/actions/calendar_action.py
"""CALENDAR intent (slice 1: read-only). Reads the configured calendar and speaks
a summary. The backend is chosen by the factory (demo / CalDAV), so this file
never mentions a specific provider."""
from datetime import datetime, timedelta

import caldav.lib.error

from actions.alarm_action import resolve
from intent.dateparse import parse_query
from intent.eventparse import parse_event, parse_delete, is_create, is_delete
from integrations.calendar_factory import get_calendar_source
from integrations.calendar_source import Event
from utils import confirm

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


def _handle_create(source, ctx, now):
    """Build the event, then ASK before writing it (see utils/confirm)."""
    request = parse_event(ctx.query)
    if not request["title"]:
        return "What should I call the event?"
    if request["time"] is None:
        return "When should I schedule that?"

    start, _hour24 = resolve(now, request["time"])
    event = Event(
        start=start,
        end=start + timedelta(minutes=request["duration_minutes"]),
        summary=request["title"].capitalize(),
    )

    def create():
        try:
            source.create_event(event)
        except _CONN_ERRORS:
            return "I couldn't reach your calendar."
        return f"Added {event.summary} to your calendar."

    confirm.request(create)
    # State it back in full so a mis-transcription is obvious before it's written.
    return f"Shall I add {event.summary} {_speak_when(event, now)}?"


# How far ahead a "remove X" request looks. Long enough to find next week's
# dentist appointment; the confirmation says the date back, so a wrong week is
# caught by ear before anything is deleted.
_DELETE_HORIZON_DAYS = 14


def _best_match(events, wanted):
    """The event whose summary best matches the words the person said.

    Word overlap, not fuzzy matching — same reasoning as smart_home's device
    matching. Returns None when nothing shares a word, because deleting the
    "closest" thing to a sentence we didn't understand is exactly the failure
    the confirmation gate exists to prevent.
    """
    words = set(wanted.lower().split())
    if not words:
        return None
    best, best_score = None, 0
    for event in events:
        score = len(words & set(event.summary.lower().split()))
        if score > best_score:
            best, best_score = event, score
    return best


def _handle_delete(source, ctx, now):
    """Find the event they named, then ASK before removing it."""
    title = parse_delete(ctx.query)["title"]
    if not title:
        return "Which event should I remove?"

    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        events = source.events_between(start, start + timedelta(days=_DELETE_HORIZON_DAYS))
    except _CONN_ERRORS:
        return "I couldn't reach your calendar."

    event = _best_match(events, title)
    if event is None:
        return f"I couldn't find {title} on your calendar."

    def remove():
        try:
            source.delete_event(event)
        except _CONN_ERRORS:
            return "I couldn't reach your calendar."
        except LookupError:
            return f"{event.summary} wasn't there any more."
        return f"Removed {event.summary} from your calendar."

    confirm.request(remove)
    # Say it back in full — the title, the day AND the time — so a mishearing is
    # obvious while it can still be stopped.
    return f"Shall I remove {event.summary}, {_speak_when(event, now)}?"


def handle(slots, ctx):
    source = get_calendar_source()
    if source is None:
        return "Your calendar isn't set up yet."
    now = datetime.now()
    # Delete is checked BEFORE create: "cancel my 3pm" contains no create verb,
    # but "take the dentist off my calendar" would trip "put"/"make" in a
    # longer sentence, and removing something is the less recoverable of the two.
    if is_delete(ctx.query):
        return _handle_delete(source, ctx, now)
    if is_create(ctx.query):
        return _handle_create(source, ctx, now)
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
