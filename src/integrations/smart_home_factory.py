# src/integrations/smart_home_factory.py
"""Pick the smart-home backend from the environment.

One place decides which source the handler gets, so `smart_home_action` never
mentions Home Assistant. Config comes from a gitignored `.env` at the repo root:

    JANET_SMART_HOME=demo          # use the in-memory fake (demos / testing)
    JANET_HA_URL=http://homeassistant.local:8123
    JANET_HA_TOKEN=<long-lived access token>
"""
import os

from dotenv import load_dotenv

from integrations.demo_home import demo_devices
from integrations.homeassistant_source import HomeAssistantSource
from integrations.smart_home_source import FakeSmartHome

# Reads .env into os.environ once, at import. Real environment variables win, so
# `JANET_SMART_HOME=demo python ...` still overrides the file.
load_dotenv()


def get_smart_home_source():
    """FakeSmartHome when JANET_SMART_HOME=demo, HomeAssistantSource when a URL
    and token are configured, otherwise None (nothing is set up)."""
    if os.getenv("JANET_SMART_HOME", "").strip().lower() == "demo":
        return FakeSmartHome(demo_devices())

    url = os.getenv("JANET_HA_URL", "").strip()
    token = os.getenv("JANET_HA_TOKEN", "").strip()
    if url and token:
        return HomeAssistantSource(url, token)

    # Both halves are required — a URL with no token would only ever 401.
    return None
