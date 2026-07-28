# src/actions/weather_action.py
"""WEATHER intent handler — speaks today's or tomorrow's weather.

Reads the RAW transcript on ctx.query (not the normalized form) because place
names arrive capitalised there — "London", not "london" — which is what we want
to say back. Everything the handler needs is in that one sentence, so WEATHER
carries no slots.
"""
import re

from integrations.weather_factory import get_weather_source
from integrations.weather_source import WeatherUnavailable, LocationNotFound

# Spoken when JANET_WEATHER* isn't configured — a setup problem, not a failure.
NOT_CONFIGURED = "Your location isn't set up yet."
# Spoken when the service/network is down. Operational, so we stay graceful.
UNAVAILABLE = "I couldn't get the weather right now."
# Spoken when the geocoder doesn't recognise the place we heard.
UNKNOWN_PLACE = "I couldn't find that place."

# Words that follow "in"/"for" but are NOT a place. Without this, "what's the
# weather in the morning" would try to geocode "the morning".
_NOT_PLACES = {
    "the", "a", "an", "my", "this", "that", "here", "there",
    "today", "tonight", "tomorrow", "now", "later",
    "morning", "afternoon", "evening", "night",
    "week", "weekend", "celsius", "fahrenheit", "degrees",
}

# Place names are short — "New York City" is already 3 words. Capping keeps a
# rambling sentence from becoming a 10-word "city".
_MAX_PLACE_WORDS = 3


def parse_location(query):
    """Pull an explicit place out of "weather in London" — or None. Pure.

    Every "in"/"for"/"at" is tried in turn, because the first one often isn't
    the place: "the weather for tomorrow in New York" must find New York.
    """
    for match in re.finditer(r"\b(?:in|for|at)\b", query, re.IGNORECASE):
        # everything after this preposition, up to the next punctuation
        tail = re.split(r"[,.?!]", query[match.end():])[0]
        words = []
        for word in tail.split():
            word = word.strip(".,?!'\"")
            if word.lower() in _NOT_PLACES:
                break                                    # not a place — stop here
            words.append(word)
            if len(words) == _MAX_PLACE_WORDS:
                break
        if words:
            # Title-case only affects how it's SPOKEN ("london" -> "London"); the
            # geocoder is case-insensitive and returns its own canonical name.
            return " ".join(words).title()
    return None


def parse_day_offset(query):
    """0 = today, 1 = tomorrow. Only these two days for now. Pure."""
    return 1 if "tomorrow" in query.lower() else 0


def _round(celsius):
    """Whole degrees — "22.4 degrees" sounds like a lab reading, not a forecast."""
    return int(round(celsius))


def _article(percent):
    """"a" or "an" for a spoken percentage — "an 80 percent chance", but "a 60
    percent chance". Only 8, 11, 18 and the eighties start with a vowel sound."""
    return "an" if str(percent).startswith(("8", "11", "18")) else "a"


def speak_current(weather):
    """"It's 22 degrees and partly cloudy in Pune." """
    line = f"It's {_round(weather.temperature_c)} degrees and {weather.description}"
    if weather.location_name:
        line += f" in {weather.location_name}"
    return line + "."


def speak_forecast(weather, when="Tomorrow"):
    """"Tomorrow in Pune: partly cloudy, high 28, low 19." """
    where = f" in {weather.location_name}" if weather.location_name else ""
    parts = [weather.description]
    if weather.high_c is not None:
        parts.append(f"high {_round(weather.high_c)}")
    if weather.low_c is not None:
        parts.append(f"low {_round(weather.low_c)}")
    line = f"{when}{where}: " + ", ".join(parts) + "."
    # Only mention rain when it's actually likely — a 10% chance isn't worth a
    # second sentence out loud.
    if weather.precipitation_chance is not None and weather.precipitation_chance >= 50:
        chance = weather.precipitation_chance
        # Open-Meteo reports "precipitation", which is a mouthful out loud — say
        # the word that matches the day.
        kind = "snow" if "snow" in weather.description else "rain"
        line += f" There's {_article(chance)} {chance} percent chance of {kind}."
    return line


def handle(slots, ctx):
    """WEATHER intent entry: today's conditions, or tomorrow's outlook.
    `slots` is unused — the raw query on ctx.query carries the day and place."""
    query = ctx.query.strip()
    source = get_weather_source(city=parse_location(query))
    if source is None:
        return NOT_CONFIGURED
    day_offset = parse_day_offset(query)
    try:
        if day_offset == 0:
            return speak_current(source.current())
        return speak_forecast(source.forecast(day_offset))
    except LocationNotFound:
        return UNKNOWN_PLACE
    except WeatherUnavailable:
        # Service down / no network — operational, not a bug, so speak a line
        # instead of crashing. Anything else propagates (bugs stay visible).
        return UNAVAILABLE
