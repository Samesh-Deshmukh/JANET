# src/integrations/weather_factory.py
"""Pick the weather backend from the environment (.env), so the handler never
reads config itself. Returns None when weather isn't set up at all."""
import os

from dotenv import load_dotenv

from integrations.weather_source import FakeWeatherSource, DEMO_LOCATION
from integrations.openmeteo_source import OpenMeteoSource


def _env(name):
    """Trimmed env var, or '' — empty and unset should behave the same."""
    return os.getenv(name, "").strip()


def get_weather_source(city=None):
    """Build a WeatherSource from the environment, or None if not configured.

    Config (a gitignored `.env`, loaded here):
      JANET_WEATHER=demo             -> the fake source, no network
      JANET_WEATHER_LAT / _LON       -> Open-Meteo at those coordinates
      JANET_WEATHER_CITY=Pune        -> Open-Meteo, geocoded on first use
                                        (also names the place when LAT/LON are set)

    `city` is an explicit place from the query ("weather in London"). It
    overrides the configured location, but only once weather is configured at
    all — an unconfigured JANET should never make a surprise network call.
    """
    load_dotenv()                       # no-op if there's no .env file

    if _env("JANET_WEATHER").lower() == "demo":
        # Demo answers for any city so "weather in London" is still testable.
        return FakeWeatherSource(location_name=city or DEMO_LOCATION)

    latitude, longitude = _env("JANET_WEATHER_LAT"), _env("JANET_WEATHER_LON")
    configured_city = _env("JANET_WEATHER_CITY")
    configured = bool((latitude and longitude) or configured_city)

    if not configured:
        return None                     # -> handler says "your location isn't set up yet"

    if city:
        return OpenMeteoSource(location_name=city)   # geocoded lazily

    if latitude and longitude:
        try:
            return OpenMeteoSource(float(latitude), float(longitude),
                                   location_name=configured_city)
        except ValueError:
            # A typo'd .env shouldn't kill the assistant — say so and fall
            # through to the city name if there is one.
            print("⚠  JANET_WEATHER_LAT/LON are not numbers — ignoring them")

    if configured_city:
        return OpenMeteoSource(location_name=configured_city)
    return None
