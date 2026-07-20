# src/integrations/demo_calendar.py
"""A fake calendar seeded relative to `now`, for demos and end-to-end testing.
Selected with JANET_CALENDAR=demo. Not a real calendar."""
from datetime import datetime, timedelta, time as dtime

from integrations.calendar_source import Event


def demo_events(now):
    """Sample events around `now` so a demo always has something upcoming."""
    today = now.date()

    def at(d, h, m=0):
        return datetime.combine(d, dtime(h, m))

    tomorrow = today + timedelta(days=1)
    friday = today + timedelta(days=(4 - today.weekday()) % 7)   # next Friday (or today)
    return [
        Event(at(today, 9, 0), at(today, 9, 30), "Standup"),
        Event(at(today, 12, 30), at(today, 13, 30), "Lunch with Alex", location="Cafe"),
        Event(at(tomorrow, 15, 0), at(tomorrow, 16, 0), "Dentist"),
        Event(at(friday, 10, 0), at(friday, 11, 0), "Project review"),
        Event(at(today, 0, 0), at(tomorrow, 0, 0), "Company holiday", all_day=True),
    ]
