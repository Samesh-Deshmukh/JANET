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


def get_calendar_source():
    """Return the configured source, or None when nothing is set up."""
    if os.environ.get("JANET_CALENDAR", "").lower() == "demo":
        # The fake calendar: sample events seeded around "now", no server needed.
        return FakeCalendarSource(demo_events(datetime.now()))
    url = os.environ.get("JANET_CALDAV_URL")
    user = os.environ.get("JANET_CALDAV_USER")
    password = os.environ.get("JANET_CALDAV_PASSWORD")
    if url and user and password:
        return CalDAVSource(url, user, password)
    return None
