# src/actions/alarm_action.py
"""Schedule spoken alarms from a parsed spec. Split into pure clock math
(resolve / next_occurrence / formatting) and the stateful scheduler + handler."""
from datetime import datetime, timedelta, time as dtime

from intent.timeparse import ALL_DAYS, WEEKDAY_SET, WEEKEND_SET

_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _candidate_hours(hour, meridiem):
    """The 24-hour hour(s) to consider for a spoken time. Bare 1-12 with no
    meridiem yields both readings so the soonest-future one can win."""
    if meridiem == "am":
        return [0 if hour == 12 else hour]
    if meridiem == "pm":
        return [12 if hour == 12 else hour + 12]
    if hour >= 13:
        return [hour]
    return [0 if hour == 12 else hour, 12 if hour == 12 else hour + 12]


def next_occurrence(after, hour24, minute, weekdays):
    """Soonest datetime strictly after `after` at hour24:minute whose weekday is
    in `weekdays`. weekdays is non-empty, so a match exists within 7 days."""
    for day in range(8):
        d = (after + timedelta(days=day)).date()
        if d.weekday() in weekdays:
            cand = datetime.combine(d, dtime(hour24, minute))
            if cand > after:
                return cand
    raise ValueError("no occurrence found")   # unreachable: weekdays non-empty


def resolve(now, spec):
    """(target_datetime, hour24) for a set spec. Applies the resolution rule:
    soonest-future meridiem inference, specific weekdays, and the tomorrow floor."""
    if spec["tomorrow"]:
        after = datetime.combine((now + timedelta(days=1)).date(), dtime(0, 0))
    else:
        after = now
    best, best_hour = None, None
    for hour24 in _candidate_hours(spec["hour"], spec["meridiem"]):
        cand = next_occurrence(after, hour24, spec["minute"], spec["weekdays"])
        if best is None or cand < best:
            best, best_hour = cand, hour24
    return best, best_hour


def speak_time(hour24, minute):
    """'7 AM' or '7:30 PM' — spoken form of a 24-hour time."""
    h12 = hour24 % 12 or 12
    ap = "AM" if hour24 < 12 else "PM"
    return f"{h12}:{minute:02d} {ap}" if minute else f"{h12} {ap}"


def recurrence_phrase(weekdays):
    """Human phrase for a recurring alarm's day set."""
    if weekdays == ALL_DAYS:
        return "every day"
    if weekdays == WEEKDAY_SET:
        return "every weekday"
    if weekdays == WEEKEND_SET:
        return "on weekends"
    if len(weekdays) == 1:
        (d,) = tuple(weekdays)
        return f"on {_NAMES[d]}s"
    return "on " + ", ".join(_NAMES[d] for d in sorted(weekdays))
