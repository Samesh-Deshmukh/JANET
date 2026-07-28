# src/integrations/openmeteo_source.py
"""Open-Meteo-backed WeatherSource. Free, no API key, no account.

Split on purpose: `to_current` / `to_forecast` are PURE (JSON dict in, Weather
out) so they can be tested offline against a sample payload; only `_get` and the
`OpenMeteoSource` methods touch the network.
"""
import requests

from integrations.weather_source import Weather, WeatherUnavailable, LocationNotFound

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Short: a voice assistant that hangs for 30s feels broken. Better to say
# "I couldn't get the weather right now" quickly.
TIMEOUT_S = 6

# WMO weather interpretation codes (the `weather_code` field) -> short spoken
# words. Kept as a flat table because that's the easiest thing to read and edit;
# every value is chosen to sound natural after "and ..." or "Tomorrow: ...".
WEATHER_CODES = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "cloudy",
    45: "foggy",
    48: "foggy",
    51: "drizzly",
    53: "drizzly",
    55: "drizzly",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "rainy",
    63: "rainy",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "snowy",
    73: "snowy",
    75: "heavy snow",
    77: "snowy",
    80: "showery",
    81: "showery",
    82: "heavy showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorms",
    96: "thunderstorms",
    99: "thunderstorms",
}

# Spoken when a code isn't in the table (Open-Meteo could add one).
UNKNOWN_DESCRIPTION = "unsettled"

# The exact fields we ask for. Named here so the parsing functions below and the
# request stay obviously in sync.
CURRENT_FIELDS = "temperature_2m,weather_code"
DAILY_FIELDS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"


def describe_code(code):
    """WMO code -> short spoken description."""
    return WEATHER_CODES.get(code, UNKNOWN_DESCRIPTION)


def _day_value(daily, key, index):
    """One day's value out of a `daily` block, or None if the API left it out.
    Open-Meteo returns parallel arrays (one entry per day), and
    precipitation_probability_max is missing for some locations."""
    values = daily.get(key) or []
    if index < len(values):
        return values[index]
    return None


def to_current(payload, location_name):
    """PURE: Open-Meteo JSON -> the Weather right now. No network.

    Today's high/low/rain-chance come from day 0 of the `daily` block, so a
    "right now" reading can still mention the rest of the day.
    """
    current = payload["current"]
    daily = payload.get("daily", {})
    return Weather(
        location_name=location_name,
        temperature_c=current["temperature_2m"],
        description=describe_code(current["weather_code"]),
        high_c=_day_value(daily, "temperature_2m_max", 0),
        low_c=_day_value(daily, "temperature_2m_min", 0),
        precipitation_chance=_day_value(daily, "precipitation_probability_max", 0),
    )


def to_forecast(payload, location_name, day_offset):
    """PURE: Open-Meteo JSON -> one day's Weather (0 = today, 1 = tomorrow)."""
    daily = payload["daily"]
    high = _day_value(daily, "temperature_2m_max", day_offset)
    low = _day_value(daily, "temperature_2m_min", day_offset)
    code = _day_value(daily, "weather_code", day_offset)
    if code is None:
        # The API didn't return that far ahead — operational, not a bug.
        raise WeatherUnavailable(f"no forecast for day +{day_offset}")
    return Weather(
        location_name=location_name,
        temperature_c=(high + low) / 2,   # a whole day has no single temperature;
                                          # the midpoint is an honest stand-in
        description=describe_code(code),
        high_c=high,
        low_c=low,
        precipitation_chance=_day_value(daily, "precipitation_probability_max", day_offset),
    )


def _get(url, params):
    """One HTTP GET returning parsed JSON. Network failures (DNS, timeout,
    5xx, ...) become WeatherUnavailable so callers stay backend-agnostic."""
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise WeatherUnavailable(str(exc)) from exc


def geocode(city):
    """City name -> (latitude, longitude, resolved_name). Needs the network.
    Raises LocationNotFound when the geocoder has never heard of the place."""
    payload = _get(GEOCODE_URL, {"name": city, "count": 1, "format": "json"})
    results = payload.get("results") or []
    if not results:
        raise LocationNotFound(city)
    top = results[0]
    return top["latitude"], top["longitude"], top.get("name", city)


class OpenMeteoSource:
    """A WeatherSource for one place. Give it coordinates, or just a city name —
    in which case it geocodes lazily on the first request (so building one in the
    factory at startup never touches the network)."""

    def __init__(self, latitude=None, longitude=None, location_name=""):
        self._latitude = latitude
        self._longitude = longitude
        self.location_name = location_name

    def _coords(self):
        """Coordinates for this place, geocoding + caching them on first use."""
        if self._latitude is None or self._longitude is None:
            if not self.location_name:
                raise WeatherUnavailable("no coordinates and no place name")
            lat, lon, name = geocode(self.location_name)
            self._latitude, self._longitude, self.location_name = lat, lon, name
        return self._latitude, self._longitude

    def _fetch(self):
        """One forecast call covering both `current` and the daily outlook, so
        current() and forecast() are a single request each."""
        latitude, longitude = self._coords()
        return _get(FORECAST_URL, {
            "latitude": latitude,
            "longitude": longitude,
            "current": CURRENT_FIELDS,
            "daily": DAILY_FIELDS,
            "timezone": "auto",       # daily rows align with the location's own days
        })

    def current(self):
        return to_current(self._fetch(), self.location_name)

    def forecast(self, day_offset):
        return to_forecast(self._fetch(), self.location_name, day_offset)
