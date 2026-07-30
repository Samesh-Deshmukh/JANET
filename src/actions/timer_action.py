# src/actions/timer_action.py
"""Countdown timers — set one, ask what's left, cancel them.

Same shape as alarm_action/reminder_action: pure phrasing helpers first, then a
small stateful scheduler (a `Countdown` object + a module registry) and the
handler. Several timers can run at once; each remembers its length, its optional
name, and when it fires, which is what makes "how much time is left?" answerable.

> Timers are SAVED (utils/store) and re-armed at startup; one that expired while
> JANET was off is dropped, not announced late. Cancel is still all-or-nothing
> (there is no per-timer cancel yet).
"""
import time
from datetime import datetime, timedelta
from threading import Timer

from intent.timerparse import parse_timer
from utils import store

_timers = []            # running Countdown objects — the registry query + cancel need

# Largest unit first, so 90 seconds is said as "1 minute and 30 seconds".
_UNITS = (("hour", 3600), ("minute", 60), ("second", 1))


def _spoken_duration(seconds, plural=True):
    """'5 minutes' / '1 minute and 30 seconds' — a length, said out loud.

    `plural=False` gives the adjective form that goes in FRONT of a noun
    ("your 5 minute timer"), where English drops the -s.
    """
    seconds = int(seconds)
    parts = []
    for unit, size in _UNITS:
        count, seconds = divmod(seconds, size)
        if count:
            word = unit if (count == 1 or not plural) else unit + "s"
            parts.append(f"{count} {word}")
    if not parts:
        return "0 seconds" if plural else "0 second"
    if len(parts) == 1:
        return parts[0]
    # Comma-then-"and" gives the TTS a natural pause when there are three parts.
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _remaining_phrase(countdown):
    """'3 minutes and 20 seconds left', plus ' on the pasta timer' when named."""
    left = countdown.remaining()
    amount = "less than a second" if left < 1 else _spoken_duration(left)
    phrase = f"{amount} left"
    if countdown.label:
        phrase += f" on the {countdown.label} timer"
    return phrase


class Countdown:
    """One running timer: how long it was set for, its optional name, when it
    fires, and the background thread that will announce it."""

    def __init__(self, seconds, label=None, remaining=None):
        self.seconds = int(seconds)
        self.label = label
        # monotonic() is a clock that only ever counts up. Unlike the wall clock
        # it can't jump (NTP corrections, DST), so "time left" stays honest.
        #
        # `remaining` is only for restoring a saved timer: monotonic() means
        # nothing across a restart, so the saved WALL-clock deadline is turned
        # back into a remaining duration by the caller. The original `seconds`
        # is kept regardless, because that is what gets announced ("your 5
        # minute timer is done") — it should not shrink just because JANET was
        # restarted halfway through.
        self.fires_at = time.monotonic() + (self.seconds if remaining is None
                                            else max(remaining, 0))
        # Wall-clock deadline, saved so a restart can work out what is left.
        self.due = datetime.now() + timedelta(seconds=self.remaining())
        self.timer = None

    def arm(self, ctx):
        delay = max(self.fires_at - time.monotonic(), 0)
        # daemon so a running timer never blocks Ctrl-C / shutdown.
        self.timer = Timer(delay, self.fire, args=(ctx,))
        self.timer.daemon = True
        self.timer.start()

    def remaining(self):
        """Whole seconds left, never negative. Rounded (not truncated) so a
        just-set 5 minute timer answers "5 minutes", not "4 minutes 59 seconds"."""
        return max(round(self.fires_at - time.monotonic()), 0)

    def as_dict(self):
        return {"seconds": self.seconds, "label": self.label,
                "due": self.due.isoformat()}

    def fire(self, ctx):
        """Announce the timer, then de-register — timers are one-shot."""
        # Say the length (and the name, if it has one) so it's obvious WHICH
        # timer just went off when several are running.
        name = _spoken_duration(self.seconds, plural=False)
        if self.label:
            name = f"{name} {self.label}"
        ctx.speak(f"Your {name} timer is done!")
        if self in _timers:
            _timers.remove(self)
        _save()


def _save():
    store.save("timers", [c.as_dict() for c in _timers])


def restore(ctx):
    """Re-arm saved timers at startup; drop any that expired while JANET was off.

    Unlike a reminder, an expired countdown is not worth announcing late: "your
    5 minute timer is done" is information about a moment that has passed, and
    the pasta is either fine or it isn't.
    """
    saved = store.load("timers", []) or []
    now = datetime.now()
    expired = 0
    for item in saved:
        try:
            due = datetime.fromisoformat(item["due"])
            left = (due - now).total_seconds()
            if left <= 0:
                expired += 1
                continue
            countdown = Countdown(item["seconds"], item.get("label"), remaining=left)
            countdown.arm(ctx)
            _timers.append(countdown)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"\u26a0  skipping a saved timer ({exc})")
    if _timers or expired:
        print(f"\u23f2  Restored {len(_timers)} timer(s)" +
              (f", {expired} expired while JANET was off" if expired else ""))
    _save()
    return expired


def _cancel_all():
    if not _timers:
        return "You don't have any timers to cancel."
    count = len(_timers)
    for countdown in _timers:
        if countdown.timer:
            countdown.timer.cancel()    # stops the scheduled firing
    _timers.clear()
    _save()
    return "Cancelled the timer." if count == 1 else f"Cancelled {count} timers."


def _list_all():
    if not _timers:
        return "You don't have any timers running."
    items = [_remaining_phrase(c) for c in _timers]
    if len(items) == 1:
        return f"You have {items[0]}."
    # "a, b, and c" — the comma before "and" gives the TTS a natural pause.
    listed = ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"You have {len(items)} timers: {listed}."


def handle(slots, ctx):
    """TIMER intent entry: set / query remaining / cancel."""
    # Read the RAW transcript (like calc_action and reminder_action) instead of
    # relying on the slot: only the full sentence tells us WHICH request this is
    # ("set a timer" vs "how long is left" vs "cancel it").
    request = parse_timer(ctx.query or "")
    if request["action"] == "cancel":
        return _cancel_all()
    if request["action"] == "query":
        return _list_all()
    if request["action"] == "unsupported":
        return "I can only set, check, and cancel timers for now."

    # Duration: our own parse of the raw transcript FIRST — it understands spoken
    # numbers ("five minutes"), "half an hour", and combined lengths ("1 minute
    # 30 seconds"), none of which intent._slots_for's simpler regex handles. The
    # slot stays as a fallback for anything we miss. Note it defaults to 300
    # whenever ITS regex finds nothing, so a duration-less request arrives here
    # as 5 minutes rather than as "unknown" — reading the query ourselves is what
    # lets TIMER improve without editing the shared intent.py.
    seconds = request["seconds"] or slots.get("duration")
    if not seconds:
        return "How long should the timer be?"

    countdown = Countdown(seconds, request["label"])
    countdown.arm(ctx)
    _timers.append(countdown)
    _save()                     # so it survives a restart
    spoken = _spoken_duration(countdown.seconds)
    if countdown.label:
        return f"{countdown.label.capitalize()} timer set for {spoken}."
    return f"Timer set for {spoken}."
