#!/usr/bin/env python3
"""Smart-home on/off checks — the SMART_HOME path on its own, in detail.

Run it from anywhere:

    venv/bin/python tools/smart_home_check.py

Uses the demo backend, so it needs no Home Assistant, no bulbs and no network.
Unlike `smoke.py` (which asks "is JANET still alive?" across the whole pipeline),
this one goes deep on a single feature: every phrasing, the state actually
flipping, device-name matching, and the safety exclusions.

Why it exists: real bulbs are on the way, and the point of testing before they
arrive is to know which failures are JANET's and which are the hardware's. When
a bulb misbehaves later, run this first — if it passes, the fault is outside
this repo.

Section 5 fakes a Home Assistant /api/states payload shaped like WiZ bulbs, so
the adapter that will meet real hardware is exercised offline too.

Exit code 0 = everything passed. Non-zero = the number of failures.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# In-memory fake devices: believable data, no credentials, no network, no bulbs.
os.environ.setdefault("JANET_SMART_HOME", "demo")

_results = []


def check(name, got, want):
    """Compare and record. Prints what came back either way — when this is run
    against real hardware the actual value is the interesting part, not the
    pass/fail."""
    ok = got == want
    _results.append((name, ok, f"got {got!r}, want {want!r}"))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"          -> {got!r}" if ok else f"          got  {got!r}\n          want {want!r}")
    return ok


def main():
    from actions import smart_home_action as sh
    from integrations.homeassistant_source import _to_devices
    from utils.context import Context

    def say(text):
        """One utterance through the REAL handler, demo backend behind it."""
        return sh.handle({}, Context(speak=lambda *a, **k: None, query=text))

    print("JANET smart-home check\n" + "=" * 60)

    # --- 1. the phrasings people actually use -----------------------------
    # The classifier only says "SMART_HOME"; which device and what to do live
    # in the words, so parse_command carries the whole burden here.
    print("\n[parsing]")
    for text, want in [
        ("turn on the kitchen lights", ("on", "kitchen lights")),
        ("switch off the fan", ("off", "fan")),
        ("turn the desk lamp on", ("on", "desk lamp")),      # particle at the end
        ("put on the coffee maker", ("on", "coffee maker")),
        ("is the fan on", ("state", "fan")),
        ("are the kitchen lights off", ("state", "kitchen lights")),
        ("janet can you turn on the bedroom lamp please", ("on", "bedroom lamp")),
        ("what is the weather", (None, "")),                 # not a switching command
    ]:
        check(text, sh.parse_command(text), want)

    # --- 2. does the state actually flip, and stay flipped? ---------------
    # The backend is cached deliberately (a fresh one each turn would forget
    # the light you just switched on), so this also proves the cache works.
    print("\n[state]")
    check("living room lights start off",
          say("is the living room lights on"), "The living room lights are off.")
    check("turn them on",
          say("turn on the living room lights"), "Turned on the living room lights.")
    check("...and it stuck",
          say("is the living room lights on"), "The living room lights are on.")
    check("turn them off again",
          say("turn off the living room lights"), "Turned off the living room lights.")
    check("...back to off",
          say("is the living room lights on"), "The living room lights are off.")

    # --- 3. which device did you mean? ------------------------------------
    # Rare words are worth more than common ones, so "bedroom" beats "lights".
    # Only interesting with several similarly-named devices — which is exactly
    # what a house full of "... Lamp" looks like.
    print("\n[matching]")
    check("'bedroom lamp' must not grab 'Living Room Lights'",
          say("turn on the bedroom lamp"), "Turned on the bedroom lamp.")
    check("'desk lamp' resolves to the desk lamp",
          say("turn on the desk lamp"), "Turned on the desk lamp.")

    # KNOWN FAILURE, left in on purpose (2026-09-05). Asking for a device that
    # does not exist switches on a real one: "greenhouse" is in no device name,
    # but the generic "lights" alone clears match_device's `best_score > 0` bar,
    # so the first light wins. match_device's docstring claims an unknown device
    # returns None "instead of a random light" — the code diverged from that.
    # Deferred until the bulbs arrive so the fix can be verified on hardware.
    # Proposed fix: if a non-filler word appears in NO device name, return None.
    check("unknown device is refused", say("turn on the greenhouse lights"), sh.NOT_FOUND)

    print("\n[declining]")
    check("not a switching command at all", say("hello there janet"), sh.UNCLEAR)

    # --- 4. the safety exclusions -----------------------------------------
    # No wake word means a misheard sentence must never reach a lock. The
    # payload below is shaped like Home Assistant's /api/states for two WiZ
    # bulbs, their SpaceSense occupancy sensor, and a front door.
    print("\n[safety]")
    devices = _to_devices([
        {"entity_id": "light.wiz_rgbw_tunable_a1b2c3", "state": "off",
         "attributes": {"friendly_name": "Bedroom Lamp", "brightness": 254,
                        "supported_color_modes": ["color_temp", "hs"]}},
        {"entity_id": "light.wiz_rgbw_tunable_d4e5f6", "state": "on",
         "attributes": {"friendly_name": "Desk Lamp", "brightness": 128}},
        {"entity_id": "binary_sensor.wiz_rgbw_tunable_a1b2c3_occupancy", "state": "on",
         "attributes": {"friendly_name": "Bedroom Lamp Occupancy",
                        "device_class": "occupancy"}},
        {"entity_id": "lock.front_door", "state": "locked",
         "attributes": {"friendly_name": "Front Door"}},
    ])
    check("both WiZ bulbs become controllable lights",
          [(d.name, d.state, d.domain) for d in devices],
          [("Bedroom Lamp", "off", "light"), ("Desk Lamp", "on", "light")])
    check("the SpaceSense occupancy sensor is filtered out",
          any("occupancy" in d.entity_id for d in devices), False)
    check("the lock is filtered out",
          any(d.domain == "lock" for d in devices), False)

    # --- report ----------------------------------------------------------
    failures = [(n, d) for n, ok, d in _results if not ok]
    print("\n" + "=" * 60)
    print(f"{len(_results) - len(failures)}/{len(_results)} passed")
    for name, detail in failures:
        print(f"  FAILED: {name}" + (f" — {detail}" if detail else ""))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
