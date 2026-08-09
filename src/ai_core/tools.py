# src/ai_core/tools.py
"""Read-only tools the LLM can call when it needs context it wasn't given.

The classifier routes an utterance to ONE action, which is usually right. But
sometimes the answer needs something else — you ask about the weather and then
say "what about Delhi?", or JANET needs tomorrow's calendar to answer a question
about today. These tools let it go and fetch that itself.

Two deliberate design choices:

**Tools return DATA, not speech.** "Delhi: 31C, sunny" — not "It's 31 degrees and
sunny in Delhi." The whole point of ai_core/responder is that one voice does the
talking, so a tool that phrased things nicely would be phrasing them twice. Terse
facts also cost fewer tokens, and the loop can run several of them.

**Tools are READ-ONLY.** Nothing here creates an event, sends mail, sets an alarm
or touches a light. JANET has no wake word and Whisper mishears constantly, so
the model may look things up freely, but changing the world stays on the
classifier → handler → confirm path where the user is asked first.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

# Placeholders the model invents when a required argument wasn't actually said —
# "will it rain tomorrow" came back as city="default_city". Treat all of these as
# "no value given" rather than trying to look them up.
_UNSET = {"", "default_city", "default", "none", "null", "here", "current",
          "current_location", "my location", "unknown", "user_location",
          # "home" is the one the tool DESCRIPTION invites — it literally says
          # "omit for home", so the model helpfully passes city="home". Live,
          # that geocoded to Homyel in Belarus and JANET reported the weather
          # there twice with total confidence. A placeholder that names a real
          # place is worse than one that doesn't: "default_city" fails loudly,
          # "home" fails as a plausible wrong answer.
          "home", "my home", "house", "local"}


def _given(args, key):
    """The argument the model supplied, or None if it didn't really supply one."""
    value = str(args.get(key) or "").strip()
    return None if value.lower() in _UNSET else value


@dataclass
class Tool:
    name: str
    description: str          # the model picks by THIS, not by the name alone
    run: Callable             # (args: dict) -> str of facts


def _weather(args):
    from integrations.weather_factory import get_weather_source
    from integrations.weather_source import WeatherUnavailable, LocationNotFound

    city = _given(args, "city")
    source = get_weather_source(city)
    if source is None:
        return "weather: not configured"
    day = (_given(args, "day") or "today").lower()
    try:
        if "tomorrow" in day:
            w = source.forecast(1)
            return (f"weather {w.location_name} tomorrow: {w.description}, "
                    f"high {w.high_c}C, low {w.low_c}C")
        w = source.current()
        return f"weather {w.location_name} now: {w.temperature_c}C, {w.description}"
    except LocationNotFound:
        return f"weather: no such place as {city!r}"
    except WeatherUnavailable:
        return "weather: service unreachable"


def _calendar(args):
    from intent.dateparse import parse_query
    from integrations.calendar_factory import get_calendar_source

    source = get_calendar_source()
    if source is None:
        return "calendar: not configured"
    # dateparse already understands "today"/"tomorrow"/a weekday, so feed the
    # day argument straight in rather than writing a second date parser.
    request = parse_query(_given(args, "day") or "today", datetime.now())
    try:
        events = source.events_between(request["start"], request["end"])
    except Exception:                      # noqa: BLE001 - any backend failure
        return "calendar: unreachable"
    if not events:
        return f"calendar {request['label']}: nothing"
    parts = [f"{e.summary} {'all day' if e.all_day else e.start.strftime('%H:%M')}"
             for e in events[:6]]
    return f"calendar {request['label']}: " + "; ".join(parts)


def _time(args):
    return f"time now: {datetime.now():%H:%M}"


def _date(args):
    return f"date today: {datetime.now():%A %Y-%m-%d}"


def _email(args):
    from integrations.email_factory import get_email_source
    from integrations.email_source import EmailUnavailable

    source = get_email_source()
    if source is None:
        return "email: not configured"
    try:
        unread = source.unread_count()
        recent = source.recent(3)
    except EmailUnavailable:
        return "email: unreachable"
    if not recent:
        return f"email: {unread} unread, inbox empty"
    parts = [f"{m.sender} re {m.subject}" for m in recent]
    return f"email: {unread} unread; recent: " + "; ".join(parts)


def _timers(args):
    from actions import timer_action

    if not timer_action._timers:
        return "timers: none running"
    parts = []
    for countdown in timer_action._timers:
        left = int(countdown.remaining())
        label = f"{countdown.label} " if countdown.label else ""
        parts.append(f"{label}{left // 60}m{left % 60:02d}s left")
    return "timers: " + "; ".join(parts)


def _devices(args):
    from integrations.smart_home_factory import get_smart_home_source
    from integrations.smart_home_source import SmartHomeUnavailable

    source = get_smart_home_source()
    if source is None:
        return "smart home: not configured"
    try:
        devices = source.list_devices()
    except SmartHomeUnavailable:
        return "smart home: unreachable"
    wanted = _given(args, "device")
    if wanted:
        needle = wanted.lower()
        devices = [d for d in devices if needle in d.name.lower()] or devices
    return "devices: " + "; ".join(f"{d.name}={d.state}" for d in devices[:8])


# The catalogue. `description` matters more than `name`: with names alone the
# model sent "what's on my calendar tomorrow" to get_time. With these it picked
# correctly every time.
TOOLS = [
    Tool("get_weather",
         "weather for a place. args: city (optional, omit for home), day ('today' or 'tomorrow')",
         _weather),
    Tool("read_calendar",
         "the user's calendar events. args: day ('today', 'tomorrow', 'this week', or a weekday)",
         _calendar),
    Tool("get_time", "the current clock time. args: none", _time),
    Tool("get_date", "today's date and weekday. args: none", _date),
    Tool("check_email", "unread count and the most recent senders/subjects. args: none", _email),
    Tool("list_timers", "timers currently running and how long is left. args: none", _timers),
    Tool("device_state",
         "on/off state of smart home devices. args: device (optional, omit for all)",
         _devices),
]

NO_TOOL = "none"
_BY_NAME = {tool.name: tool for tool in TOOLS}


def names():
    """Every valid choice, for the response schema's enum.

    Putting these in an enum means constrained decoding makes it *impossible*
    for the model to name a tool that doesn't exist — a guarantee native
    tool-calling APIs don't give you.
    """
    return [NO_TOOL] + [tool.name for tool in TOOLS]


def catalogue():
    """The tool list as prompt text."""
    lines = [f"- {NO_TOOL}: the FACTS already answer it, or no lookup is needed"]
    lines += [f"- {tool.name}: {tool.description}" for tool in TOOLS]
    return "\n".join(lines)


def run(name, args):
    """Execute a tool and return its facts, or None if there's nothing to run."""
    tool = _BY_NAME.get(name)
    if tool is None:
        return None
    try:
        return tool.run(args or {})
    except Exception as exc:               # noqa: BLE001
        # A broken tool must not take the whole reply down — JANET should still
        # answer from what it already has. The failure is reported as facts so
        # the model can say it couldn't find out, rather than inventing.
        print(f"⚠  tool {name} failed: {type(exc).__name__}: {exc}")
        return f"{name}: failed"
