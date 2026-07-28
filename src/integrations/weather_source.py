# src/integrations/weather_source.py
"""Backend-agnostic weather model. Handlers depend on this, not on Open-Meteo."""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Weather:
    """One weather reading — either "right now" or "one whole day".

    Celsius throughout (the owner is in India). The three optional fields are
    absent for some readings/backends, so handlers must check for None before
    speaking them.
    """
    location_name: str
    temperature_c: float
    description: str                        # short spoken words: "partly cloudy"
    high_c: float | None = None
    low_c: float | None = None
    precipitation_chance: int | None = None  # percent, 0-100


class WeatherUnavailable(Exception):
    """The source couldn't answer for an *operational* reason — the service is
    down, the network is out, a request timed out. Backends translate their own
    library errors into this so handlers never import `requests`/httpx/etc.
    It is NOT for bugs: a KeyError on a malformed payload should still crash."""


class LocationNotFound(WeatherUnavailable):
    """The place name wasn't recognised by the backend's geocoder."""


class WeatherSource(Protocol):
    def current(self) -> "Weather":
        """Weather right now."""
        ...

    def forecast(self, day_offset: int) -> "Weather":
        """One day's outlook. day_offset 0 = today, 1 = tomorrow."""
        ...


# Canned values for the fake. Realistic for Pune in July: warm, wet, monsoon-ish.
DEMO_LOCATION = "Pune"
_DEMO_DAYS = [
    # today, tomorrow, day after
    Weather(DEMO_LOCATION, 22.4, "partly cloudy", 28.1, 19.3, 20),
    Weather(DEMO_LOCATION, 24.2, "rainy", 27.6, 21.2, 80),
    Weather(DEMO_LOCATION, 25.0, "clear", 30.4, 20.8, 5),
]


class FakeWeatherSource:
    """In-memory source (test double + demo). Canned but realistic values, no
    network. Selected with JANET_WEATHER=demo. Not real weather."""

    def __init__(self, location_name=DEMO_LOCATION, current=None, days=None):
        # Every argument is optional so a demo is one call, but tests can pin
        # exact numbers: FakeWeatherSource(current=Weather("Oslo", -3, "snowy")).
        self.location_name = location_name
        self._current = current or _relabel(_DEMO_DAYS[0], location_name)
        self._days = list(days) if days else [_relabel(d, location_name) for d in _DEMO_DAYS]

    def current(self):
        return self._current

    def forecast(self, day_offset):
        # Clamp instead of raising: a demo should never crash on "the weather on
        # Friday". The real backends carry a full week, this one only 3 days.
        index = min(max(day_offset, 0), len(self._days) - 1)
        return self._days[index]


def _relabel(weather, location_name):
    """Copy a canned Weather with a different place name, so a demo can answer
    'what's the weather in London' with London in the sentence."""
    return Weather(
        location_name=location_name,
        temperature_c=weather.temperature_c,
        description=weather.description,
        high_c=weather.high_c,
        low_c=weather.low_c,
        precipitation_chance=weather.precipitation_chance,
    )
