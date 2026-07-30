# src/actions/alarm_action.py
"""Schedule spoken alarms from a parsed spec. Split into pure clock math
(resolve / next_occurrence / formatting) and the stateful scheduler + handler."""
from datetime import datetime, timedelta, time as dtime

from intent.timeparse import ALL_DAYS, WEEKDAY_SET, WEEKEND_SET
from utils import store

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


from threading import Timer

_alarms = []            # active Alarm objects — the registry cancel + re-arm need


class Alarm:
    def __init__(self, hour24, minute, weekdays, recurring):
        self.hour24 = hour24
        self.minute = minute
        self.weekdays = weekdays
        self.recurring = recurring
        self.timer = None
        self.at = None          # resolved datetime, kept so it can be saved

    def arm(self, delay, ctx):
        # daemon so a pending alarm never blocks Ctrl-C / shutdown.
        self.timer = Timer(max(delay, 0), self._fire, args=(ctx,))
        self.timer.daemon = True
        self.timer.start()

    def _fire(self, ctx):
        ctx.speak(f"Alarm! It's {speak_time(self.hour24, self.minute)}.")
        if self.recurring:
            now = datetime.now()
            nxt = next_occurrence(now, self.hour24, self.minute, self.weekdays)
            self.at = nxt
            self.arm((nxt - now).total_seconds(), ctx)   # re-arm on the pinned hour
        elif self in _alarms:
            _alarms.remove(self)
        _save()

    def as_dict(self):
        return {
            "hour24": self.hour24,
            "minute": self.minute,
            "weekdays": sorted(self.weekdays),
            "recurring": self.recurring,
            # The resolved date matters for a one-shot: "7 am tomorrow" and
            # "7 am next Tuesday" are indistinguishable from hour+weekdays alone.
            "at": self.at.isoformat() if self.at else None,
        }


def _save():
    store.save("alarms", [a.as_dict() for a in _alarms])


def restore(ctx):
    """Re-arm saved alarms at startup. Returns a note for anything missed."""
    saved = store.load("alarms", []) or []
    now = datetime.now()
    missed = 0
    for item in saved:
        try:
            weekdays = set(item["weekdays"])
            alarm = Alarm(item["hour24"], item["minute"], weekdays, item["recurring"])
            if item["recurring"]:
                # A recurring alarm has no single date — just find the next one.
                target = next_occurrence(now, alarm.hour24, alarm.minute, weekdays)
            else:
                target = datetime.fromisoformat(item["at"]) if item.get("at") else None
                if target is None or target <= now:
                    # A one-shot whose moment passed while JANET was off. Saying
                    # "Alarm! It's 7 AM" at half past nine would be a lie, so it
                    # is dropped — but not silently.
                    missed += 1
                    continue
            alarm.at = target
            alarm.arm((target - now).total_seconds(), ctx)
            _alarms.append(alarm)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"⚠  skipping a saved alarm ({exc})")
    if _alarms or missed:
        print(f"⏰ Restored {len(_alarms)} alarm(s)" +
              (f", {missed} missed while JANET was off" if missed else ""))
    _save()
    return missed


def _describe(hour24, minute, spec, target):
    """The confirmation line to speak back."""
    t = speak_time(hour24, minute)
    if spec["recurring"]:
        return f"Alarm set for {t} {recurrence_phrase(spec['weekdays'])}."
    today = datetime.now().date()
    if target.date() == today:
        return f"Alarm set for {t}."
    if target.date() == today + timedelta(days=1):
        return f"Alarm set for tomorrow at {t}."
    return f"Alarm set for {t} on {target:%A}."


def _cancel_all():
    if not _alarms:
        return "There's no alarm to cancel."
    n = len(_alarms)
    for alarm in _alarms:
        if alarm.timer:
            alarm.timer.cancel()
    _alarms.clear()
    _save()
    return "Cancelled the alarm." if n == 1 else f"Cancelled {n} alarms."


def handle(slots, ctx):
    """ALARM intent entry: set (one-shot / specific-day / recurring) or cancel."""
    spec = slots["alarm"]
    if spec["action"] == "cancel":
        return _cancel_all()
    if spec["hour"] is None:
        return "What time should I set the alarm for?"
    now = datetime.now()
    target, hour24 = resolve(now, spec)
    alarm = Alarm(hour24, spec["minute"], spec["weekdays"], spec["recurring"])
    alarm.at = target
    alarm.arm((target - now).total_seconds(), ctx)
    _alarms.append(alarm)
    _save()                     # so it survives a restart
    return _describe(hour24, spec["minute"], spec, target)
