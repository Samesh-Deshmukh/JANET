# src/integrations/homeassistant_source.py
"""Home Assistant REST adapter — a SmartHomeSource backed by the owner's own
Home Assistant box on the LAN (local-first: no cloud hop, no vendor account).

Parsing (`_to_devices`) is a pure function so it can be tested offline against a
sample /api/states payload; only the two request helpers touch the network.

SAFETY — deliberate exclusion: only the domains in ALLOWED_DOMAINS
(light / switch / fan) are ever listed or acted on. Home Assistant also exposes
`lock`, `cover` (garage doors), and `alarm_control_panel`; those are filtered out
of the device list AND re-checked before any service call, so a misheard sentence
can never unlock a door.
"""
import requests

from integrations.smart_home_source import Device, SmartHomeUnavailable

# The only entity domains JANET will read or control. See the safety note above.
ALLOWED_DOMAINS = ("light", "switch", "fan")

# It's a LAN call — if it hasn't answered in 5 seconds, something is wrong and
# we'd rather say so than leave the user waiting in silence.
TIMEOUT = 5


def _to_devices(payload):
    """Turn a Home Assistant /api/states JSON list into Devices (pure — no I/O).

    Each entity looks like:
        {"entity_id": "light.living_room", "state": "on",
         "attributes": {"friendly_name": "Living Room Lights", ...}, ...}
    """
    devices = []
    for entity in payload:
        entity_id = entity.get("entity_id", "")
        domain, _, object_id = entity_id.partition(".")
        if domain not in ALLOWED_DOMAINS:
            continue                      # locks, covers, sensors, ... skipped
        attributes = entity.get("attributes") or {}
        # friendly_name is what the user called it in Home Assistant, so it's the
        # name they'll speak. Fall back to the id ("desk_lamp" -> "Desk Lamp").
        name = attributes.get("friendly_name") or object_id.replace("_", " ").title()
        raw_state = entity.get("state", "")
        # HA can also report "unavailable" / "unknown"; for a spoken answer
        # anything that isn't clearly on reads as off.
        state = "on" if raw_state == "on" else "off"
        devices.append(Device(name=name, entity_id=entity_id, state=state, domain=domain))
    return devices


class HomeAssistantSource:
    """Talks to Home Assistant's REST API with a long-lived access token
    (Home Assistant -> your profile -> Security -> Long-lived access tokens)."""

    def __init__(self, url, token):
        self._url = url.rstrip("/")       # so we can append "/api/..." safely
        self._headers = {"Authorization": f"Bearer {token}"}

    def list_devices(self):
        return _to_devices(self._get("/api/states"))

    def turn_on(self, device):
        self._call_service(device, "turn_on")

    def turn_off(self, device):
        self._call_service(device, "turn_off")

    def _call_service(self, device, service):
        """POST /api/services/<domain>/turn_on|turn_off {"entity_id": ...}"""
        if device.domain not in ALLOWED_DOMAINS:
            # Belt and braces: the list is already filtered, but a Device could
            # be built by hand. Refusing here is a bug, not an operational
            # failure, so it raises instead of returning a spoken line.
            raise ValueError(f"refusing to control {device.entity_id}")
        self._post(f"/api/services/{device.domain}/{service}",
                   {"entity_id": device.entity_id})

    def _get(self, path):
        return self._request("get", path)

    def _post(self, path, payload):
        return self._request("post", path, json=payload)

    def _request(self, method, path, **kwargs):
        """One place for the network call, so every failure mode (no route, DNS,
        timeout, 401 bad token, 500) becomes the same SmartHomeUnavailable."""
        try:
            resp = requests.request(method, self._url + path,
                                    headers=self._headers, timeout=TIMEOUT, **kwargs)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SmartHomeUnavailable(str(exc)) from exc
        return resp.json() if resp.content else None
