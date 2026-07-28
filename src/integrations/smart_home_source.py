# src/integrations/smart_home_source.py
"""Backend-agnostic smart-home model. Handlers depend on this, not on Home Assistant.

SAFETY — deliberate exclusion: this abstraction covers ONLY simple on/off devices
(lights, switches, fans). Locks, garage doors, covers, and alarm panels are
intentionally out of scope: JANET listens to a whole room with no wake word, so
anything security-sensitive must never be one misheard sentence away from opening.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Device:
    name: str          # spoken name, e.g. "Living Room Lights"
    entity_id: str     # backend id, e.g. "light.living_room"
    state: str = "off"  # "on" or "off" — what we say back and read out
    domain: str = "light"   # "light" / "switch" / "fan" — decides which service to call


class SmartHomeSource(Protocol):
    def list_devices(self) -> "list[Device]":
        """Every controllable device the backend exposes."""
        ...

    def turn_on(self, device: Device) -> None:
        ...

    def turn_off(self, device: Device) -> None:
        ...


class SmartHomeUnavailable(Exception):
    """The backend is unreachable, or rejected our token. An operational failure
    (the network / the config), not a bug — the handler catches this one and
    speaks a friendly line. It lives here, not in the Home Assistant module, so
    the handler can stay backend-agnostic."""


class FakeSmartHome:
    """In-memory source (test double + demo). State really flips, so a test can
    switch a light on and then assert the device reads "on"."""

    def __init__(self, devices):
        self._devices = list(devices)

    def list_devices(self):
        # A new list, but the SAME Device objects — callers see live state.
        return list(self._devices)

    def turn_on(self, device):
        self._set(device, "on")

    def turn_off(self, device):
        self._set(device, "off")

    def _set(self, device, state):
        """Flip the stored device's state. Matched on entity_id because the
        caller may be holding a different Device object for the same entity."""
        for known in self._devices:
            if known.entity_id == device.entity_id:
                known.state = state
        device.state = state      # keep the caller's copy in sync too
