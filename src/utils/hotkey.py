"""Global space-key detection via /dev/input (evdev).

Why this exists: the old `keyboard` library requires root on Linux, and running JANET
as root means it can't reach your (per-user) PipeWire microphone -> silent recordings.
Reading /dev/input directly only needs membership in the `input` group, so JANET can run
as your normal user and the mic works.

Setup (one time):
    sudo usermod -aG input $USER   # then log out and back in for it to take effect

Public API:
    is_pressed()  -> True while the SPACE key is held down.

A background daemon thread watches every keyboard device and keeps `_space_down` current,
so is_pressed() is just a cheap flag read.
"""
import threading
import select
import evdev
from evdev import ecodes

_space_down = False
_lock = threading.Lock()
_devices = None  # opened keyboard devices, shared with the listener thread
_listener_started = False


def _open_keyboards():
    """Open every input device that exposes the SPACE key (i.e. the real keyboards).

    Raises a clear, actionable error if /dev/input isn't readable — the usual cause is
    not being in the `input` group yet (or not having re-logged in after `usermod`).
    """
    paths = evdev.list_devices()
    if not paths:
        raise RuntimeError(
            "Can't see any /dev/input devices. Add yourself to the 'input' group:\n"
            "    sudo usermod -aG input $USER\n"
            "then log out and back in (or run `newgrp input` in this shell)."
        )

    keyboards, denied = [], 0
    for path in paths:
        try:
            dev = evdev.InputDevice(path)
        except PermissionError:
            denied += 1
            continue
        if ecodes.KEY_SPACE in dev.capabilities().get(ecodes.EV_KEY, []):
            keyboards.append(dev)

    if not keyboards:
        if denied:
            raise PermissionError(
                f"Found {denied} input device(s) but can't read any of them. "
                "You're probably not in the 'input' group yet:\n"
                "    sudo usermod -aG input $USER\n"
                "then log out/in (or `newgrp input`)."
            )
        raise RuntimeError("No keyboard with a SPACE key found in /dev/input.")
    return keyboards


def _listen():
    """Read key events from all keyboards and mirror the SPACE state into `_space_down`."""
    global _space_down
    fds = {dev.fd: dev for dev in _devices}
    while True:
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            for event in fds[fd].read():
                if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_SPACE:
                    # event.value: 1 = key down, 0 = key up, 2 = autorepeat (still held)
                    with _lock:
                        _space_down = event.value != 0


def _ensure_listener():
    """Open devices on the main thread (so errors surface to the user), then start the reader."""
    global _devices, _listener_started
    if _listener_started:
        return
    _devices = _open_keyboards()  # raises here, visibly, if perms are wrong
    threading.Thread(target=_listen, daemon=True).start()
    _listener_started = True


def is_pressed():
    """True while SPACE is held. Starts the background listener on first call."""
    _ensure_listener()
    with _lock:
        return _space_down
