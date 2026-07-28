# src/utils/volume.py
"""Read and change the system output volume through `pactl` (PipeWire).

Why `pactl`? This machine runs PipeWire, and `audio/tts.py` already plays audio
with `paplay` from the same toolkit — so `pactl` is the matching control tool.
(ALSA's `amixer` is NOT used: it talks to a different layer and doesn't reflect
the PipeWire sink our speech actually comes out of.)

Design rule for this whole file: **never raise, return None instead.** A missing
`pactl`, a weird output format, or a non-zero exit all come back as None so the
caller (actions/system_action.py) can simply speak "I couldn't change the
volume." A voice assistant should not crash because a CLI tool moved.
"""
import re
import subprocess

# pactl's own alias for "whatever the current default output device is", so we
# never have to look up a sink name (they change when you plug in headphones).
SINK = "@DEFAULT_SINK@"

# pactl answers instantly; the timeout only exists so a wedged audio server
# can't freeze the assistant's main loop forever.
TIMEOUT_SECONDS = 2


def _run(args):
    """Run `pactl <args>` and return its stdout, or None if anything went wrong.

    This is the single place where the outside world can fail, which is why all
    the error handling lives here and nowhere else in the file."""
    try:
        proc = subprocess.run(
            ["pactl", *args],
            capture_output=True,   # we need to read stdout, not print it
            text=True,             # give us str, not bytes
            timeout=TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None                # pactl not installed / audio server wedged
    if proc.returncode != 0:
        return None                # pactl ran but refused (e.g. no sink)
    return proc.stdout


def _clamp(percent):
    """Keep a level inside 0-100. We clamp ourselves rather than letting pactl
    take '150%' — PipeWire happily over-amplifies, which sounds terrible."""
    return max(0, min(100, int(percent)))


def get_volume():
    """Current output volume as a whole percent, or None if unreadable.

    pactl prints one line per channel, e.g.
        Volume: front-left: 65536 /  65% / -9.29 dB,  front-right: 65536 / 65% ...
    Both channels normally match, so we just take the FIRST percentage we see."""
    out = _run(["get-sink-volume", SINK])
    if out is None:
        return None
    match = re.search(r"(\d+)%", out)
    if match is None:
        return None                # format changed — better silent than wrong
    return int(match.group(1))


def set_volume(percent):
    """Set an absolute volume. Returns the level actually applied (after
    clamping) so the caller can speak it back, or None if pactl failed."""
    level = _clamp(percent)
    if _run(["set-sink-volume", SINK, f"{level}%"]) is None:
        return None
    return level


def change_volume(delta):
    """Change the volume by `delta` percentage points. Returns the new level, or
    None if we couldn't read or set it.

    We deliberately read-then-write instead of using pactl's relative form
    (`+10%`): that way we know the resulting number to say out loud, and our own
    0-100 clamp applies."""
    current = get_volume()
    if current is None:
        return None
    return set_volume(current + delta)


def is_muted():
    """True/False, or None if unreadable. pactl prints exactly 'Mute: yes' or
    'Mute: no', so we just look at how that line ends."""
    out = _run(["get-sink-mute", SINK])
    if out is None:
        return None
    line = out.strip().lower()
    if line.endswith("yes"):
        return True
    if line.endswith("no"):
        return False
    return None                    # unexpected format


def set_muted(muted):
    """Mute (True) or unmute (False). Returns True on success, None on failure —
    same None-means-unavailable convention as the readers above."""
    flag = "1" if muted else "0"
    if _run(["set-sink-mute", SINK, flag]) is None:
        return None
    return True
