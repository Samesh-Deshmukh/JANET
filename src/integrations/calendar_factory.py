# src/integrations/calendar_factory.py
"""Choose the configured CalendarSource from env: demo, CalDAV, or None.

Keeping the choice here means the handler never imports a specific backend —
swapping CalDAV for Google later is a change to this file only.
"""
import os
from datetime import datetime

from dotenv import load_dotenv

from integrations.calendar_source import FakeCalendarSource
from integrations.demo_calendar import demo_events
from integrations.caldav_source import CalDAVSource

load_dotenv()

# Built once and reused (like the Whisper/DistilBERT singletons). This matters
# for the demo calendar: an event you add has to still be there on the next
# question, which a fresh instance per call would throw away.
_source = None
_built = False


def get_calendar_source():
    """Return the configured source, or None when nothing is set up."""
    global _source, _built
    if not _built:
        _source = _build_source()
        _built = True
    return _source


def _build_source():
    if os.environ.get("JANET_CALENDAR", "").lower() == "demo":
        # The fake calendar: sample events seeded around "now", no server needed.
        return FakeCalendarSource(demo_events(datetime.now()))
    url = os.environ.get("JANET_CALDAV_URL")
    user = os.environ.get("JANET_CALDAV_USER")
    password = os.environ.get("JANET_CALDAV_PASSWORD")
    if url and user and password:
        return CalDAVSource(url, user, password)
    return None


def reset():
    """Drop the cached source so config changes take effect (used by tests)."""
    global _source, _built
    _source = None
    _built = False
