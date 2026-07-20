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


class CalendarSource(Protocol):
    def events_between(self, start: datetime, end: datetime) -> "list[Event]":
        """Events overlapping [start, end), sorted by start."""
        ...


class FakeCalendarSource:
    """In-memory source (test double + demo). Holds a fixed list of Events."""

    def __init__(self, events):
        self._events = list(events)

    def events_between(self, start, end):
        hits = [e for e in self._events if e.start < end and e.end > start]
        return sorted(hits, key=lambda e: e.start)
