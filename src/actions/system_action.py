# src/actions/system_action.py
"""SYSTEM intent handler — JANET controlling and reporting on herself.

Covers four things today: the machine's output **volume** (via utils/volume.py),
a spoken list of **capabilities**, **repeating** the last reply, and a short
**status** line. Anything else SYSTEM-ish gets an honest "not yet".

SAFETY — deliberate exclusion (do not "fix" this):
    There is NO code path here that shuts down, reboots, suspends, logs out, or
    kills a process, and there must never be one. Voice input is unreliable —
    Whisper `tiny` mishears constantly — so a misheard word must not be able to
    power off the machine. Those requests are recognised only so we can DECLINE
    them out loud; the handler never runs a destructive command.

Like calc_action/general_action this reads the raw transcript from `ctx.query`
rather than a slot: the phrasing itself is all the information we need, so the
intent classifier doesn't have to parse anything for us.
"""
import re
import time

from intent.normalize import normalize
from intent.numwords import words_to_numbers
from utils import volume

# How much one "volume up" / "volume down" moves the level, in percentage points.
STEP = 10

# What JANET says she can do. Spoken aloud, so it must stay ONE OR TWO SENTENCES —
# never read out a long list over voice.
#
# KEEP IN SYNC with intent/dispatch.py's REGISTRY (the real list of built
# handlers: TIME, DATE, TIMER, ALARM, CALC, GENERAL, SYSTEM). It's a plain
# constant on purpose — importing dispatch from here would be a circular import,
# since dispatch imports this module.
CAPABILITIES = (
    "I can tell you the time and date, set timers and alarms, do maths, "
    "control my volume, and answer general questions. Just talk to me, "
    "there's no wake word."
)

# Spoken when the volume system isn't available (no pactl, no audio server).
VOLUME_UNAVAILABLE = "I couldn't change the volume."

# Power/session/process words. Matching one of these gets a refusal, nothing else.
_DESTRUCTIVE = (
    "shut down", "shutdown", "power off", "power down", "turn off the computer",
    "reboot", "restart", "suspend", "hibernate",
    "log out", "logout", "log off", "kill", "sleep",
)
REFUSE_DESTRUCTIVE = (
    "I don't do power commands like shutting down, restarting, or sleeping. "
    "That's switched off on purpose."
)

_REPEAT = ("say that again", "say it again", "repeat that", "repeat it",
           "what did you say", "come again", "repeat yourself")
_CAPABILITIES = ("what can you do", "what are your capabilities",
                 "what are your commands", "what commands", "what do you do",
                 "help me out with what you can do", "what can you help with")
_STATUS = ("are you there", "are you still there", "are you awake",
           "are you listening", "are you online", "are you working",
           "your status", "status report", "how are you doing", "how are you",
           "you there", "check your systems", "run a diagnostic")

# Volume phrasings. Order of the CHECKS in `_volume_reply` matters more than these
# lists — see the comments there.
_UNMUTE = ("unmute", "un mute", "you can talk again")
_MUTE = ("mute", "be quiet", "hush", "silence yourself")
_UP = ("volume up", "turn it up", "turn up", "turn the volume up", "louder",
       "raise the volume", "raise your volume", "speak up", "increase the volume",
       "turn the sound up", "crank")
_DOWN = ("volume down", "turn it down", "turn down", "turn the volume down",
         "quieter", "softer", "lower the volume", "lower your volume",
         "quiet down", "too loud", "decrease the volume", "turn the sound down")
_VOLUME_QUERY = ("what is the volume", "whats the volume", "how loud",
                 "what volume", "your volume level")
_MUTE_QUERY = ("are you muted", "are you mute", "are you on mute",
               "is the sound muted")

# When the process started, for the uptime line. monotonic() is a steadily
# increasing counter (immune to the wall clock changing), which is exactly what
# you want for "how long have I been up".
_STARTED = time.monotonic()


def _has(text, phrases):
    """True if any phrase appears in the (already normalized) text."""
    return any(phrase in text for phrase in phrases)


def uptime_phrase(seconds):
    """'less than a minute' / '5 minutes' / '2 hours'. Pure, so it's testable
    without waiting around for real time to pass."""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "less than a minute"
    if minutes < 60:
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    hours = minutes // 60
    return "1 hour" if hours == 1 else f"{hours} hours"


def _absolute_level(text):
    """The level for an absolute request ('set the volume to 40'), else None.

    Checked BEFORE up/down so "turn it up to 60" sets 60 instead of nudging."""
    if "all the way up" in text or "to max" in text or "maximum" in text:
        return 100
    if "all the way down" in text or "to min" in text or "minimum" in text:
        return 0
    # A bare number only counts as a target when the phrasing points at one,
    # so "turn it up a notch" (no number) doesn't accidentally match.
    if "set" in text or " to " in text or " at " in text:
        match = re.search(r"\b(\d{1,3})\b", text)
        if match:
            return int(match.group(1))
    return None


def _last_reply(history):
    """JANET's most recent spoken reply from the conversation history, or None.

    Note main.py records an exchange only AFTER speaking it, so the "repeat that"
    turn isn't in there yet — we correctly get the reply before it."""
    for message in reversed(history.messages()):
        if message["role"] == "assistant":
            return message["content"]
    return None


def _volume_reply(text):
    """Handle every volume/mute phrasing, or return None if this isn't one."""
    # Questions before commands: "are you muted" is not an order to mute.
    if _has(text, _MUTE_QUERY):
        muted = volume.is_muted()
        if muted is None:
            return "I couldn't read the volume."
        return "Yes, I'm muted." if muted else "No, I'm not muted."
    # "unmute" contains "mute", so it MUST be tested before the mute command.
    if _has(text, _UNMUTE):
        return "Unmuted." if volume.set_muted(False) else VOLUME_UNAVAILABLE
    # "How loud are you" asks; it doesn't change anything.
    if _has(text, _VOLUME_QUERY):
        if volume.is_muted():
            return "I'm muted."
        level = volume.get_volume()
        if level is None:
            return "I couldn't read the volume."
        return f"The volume is {level} percent."
    if _has(text, _MUTE):
        return "Muted." if volume.set_muted(True) else VOLUME_UNAVAILABLE

    # Only the level-changing phrasings are left; all of them mention volume,
    # loudness, or an up/down command.
    if not _has(text, _UP + _DOWN) and "volume" not in text:
        return None

    level = _absolute_level(text)
    if level is None:
        if _has(text, _UP):
            level = volume.change_volume(STEP)
        elif _has(text, _DOWN):
            level = volume.change_volume(-STEP)
        else:
            return None            # mentions "volume" but asks for nothing we do
    else:
        level = volume.set_volume(level)
    if level is None:
        return VOLUME_UNAVAILABLE
    return f"Volume is now {level} percent."


def handle(slots, ctx):
    """SYSTEM intent entry. `slots` is unused — everything comes from ctx.query."""
    # Normalize (lowercase, no punctuation) so phrase matching works on Whisper's
    # raw output, then spell out spoken numbers ("fifty" -> "50") for "set the
    # volume to fifty".
    text = words_to_numbers(normalize(ctx.query or ""))

    # FIRST, and deliberately: refuse power commands. Checking this before
    # anything else means no misheard phrase can fall through into another branch.
    if _has(text, _DESTRUCTIVE):
        return REFUSE_DESTRUCTIVE

    if _has(text, _REPEAT):
        last = _last_reply(ctx.history)
        return last if last else "I haven't said anything yet."

    if _has(text, _CAPABILITIES):
        return CAPABILITIES

    if _has(text, _STATUS):
        return f"I'm running and listening. I've been up for {uptime_phrase(time.monotonic() - _STARTED)}."

    reply = _volume_reply(text)
    if reply is not None:
        return reply

    # Addressed, classified SYSTEM, but not something we've built (sleep mode,
    # microphone toggling, "what version are you", ...). Be honest.
    return "I can't do that yet."
