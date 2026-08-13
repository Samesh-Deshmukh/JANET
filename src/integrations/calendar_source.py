# src/integrations/calendar_source.py
"""Backend-agnostic calendar model. Handlers depend on this, not on CalDAV/Google."""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Event:
    start: datetime
    end: datetime
    summary: str
    all_day: bool = False
    location: str = ""
    # The backend's own identifier, when it has one. Needed to DELETE an event:
    # a summary and a start time identify an event to a person, but only the uid
    # identifies it to the server. Empty for the in-memory sources, which can
    # match on the fields themselves.
    uid: str = ""
    # Free-text body of the event, below the title. Spoken answers never read
    # this out — it is for detail a title shouldn't carry (a teacher's name, a
    # class code, a meeting link).
    description: str = ""
    # iCalendar recurrence lines, e.g.
    #   ("RRULE:FREQ=WEEKLY;BYDAY=MO;UNTIL=20270630T235959Z",)
    # A TUPLE, not a list: a dataclass field cannot have a mutable default,
    # because every Event built without one would share the same list object.
    # Empty means a one-off event, which is what everything else in JANET makes.
    recurrence: tuple[str, ...] = ()
    # Minutes before the start to be notified. None means "say nothing about
    # reminders and let the calendar's own defaults apply" — which is NOT the
    # same as 0, a real request to be notified exactly at the start time. A
    # plain `int = 0` default would silently override the user's calendar
    # settings on every event JANET has ever created.
    reminder_minutes: int | None = None


class CalendarSource(Protocol):
    def events_between(self, start: datetime, end: datetime) -> "list[Event]":
        """Events overlapping [start, end), sorted by start."""
        ...

    def create_event(self, event: "Event") -> None:
        """Add an event to the calendar."""
        ...

    def delete_event(self, event: "Event") -> None:
        """Remove an event. Raise LookupError if it is no longer there."""
        ...


class FakeCalendarSource:
    """In-memory source (test double + demo). Holds a fixed list of Events."""

    def __init__(self, events):
        self._events = list(events)

    def events_between(self, start, end):
        hits = [e for e in self._events if e.start < end and e.end > start]
        return sorted(hits, key=lambda e: e.start)

    def create_event(self, event):
        self._events.append(event)

    def delete_event(self, event):
        """Match on start + summary — there is no server to give us a uid."""
        for i, existing in enumerate(self._events):
            if existing.start == event.start and existing.summary == event.summary:
                del self._events[i]
                return
        raise LookupError(f"no event {event.summary!r} to delete")
