# src/integrations/caldav_source.py
"""CalDAV-backed CalendarSource. Parsing (ics_to_events) is pure and fixture-tested;
the live connection needs a server + credentials (owner-configured)."""
from datetime import datetime, date, time as dtime

import caldav
import caldav.lib.error
from icalendar import Calendar as iCal

from integrations.calendar_source import Event


def _as_local_naive(value):
    """Normalize an icalendar date/datetime to a naive local datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone().replace(tzinfo=None)   # to local, drop tz
        return value
    return datetime.combine(value, dtime(0, 0))               # all-day date -> midnight


def ics_to_events(ics_text):
    """Parse raw ICS text into a list of Event (pure)."""
    events = []
    cal = iCal.from_ical(ics_text)
    for comp in cal.walk("VEVENT"):
        ds = comp.get("dtstart").dt
        de = comp.get("dtend")
        all_day = isinstance(ds, date) and not isinstance(ds, datetime)
        start = _as_local_naive(ds)
        end = _as_local_naive(de.dt) if de else start
        events.append(Event(
            start=start,
            end=end,
            summary=str(comp.get("summary") or "(no title)"),
            all_day=all_day,
            location=str(comp.get("location") or ""),
        ))
    return events


class CalDAVSource:
    def __init__(self, url, username, password):
        self._url = url
        self._username = username
        self._password = password

    def _calendars(self):
        client = caldav.DAVClient(url=self._url, username=self._username, password=self._password)
        return client.principal().calendars()

    def events_between(self, start, end):
        events = []
        for calendar in self._calendars():
            for obj in calendar.search(start=start, end=end, event=True, expand=True):
                events += ics_to_events(obj.data)
        return sorted(events, key=lambda e: e.start)

    def create_event(self, event):
        """Write a new event to the first calendar on the account."""
        calendars = self._calendars()
        if not calendars:
            raise caldav.lib.error.NotFoundError("no calendar to write to")
        # add_event builds the VEVENT for us from these properties.
        calendars[0].add_event(
            dtstart=event.start,
            dtend=event.end,
            summary=event.summary,
        )
