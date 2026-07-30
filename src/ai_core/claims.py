# src/ai_core/claims.py
"""Catch JANET claiming to have done something it did not do.

Found in live testing, and it is the worst failure the system had:

    you:   "Can you turn it down to 55 again?"
    gates: classifier says NONE 49% -> rescued by the addressing check
    route: GENERAL, whose handler returns no facts at all
    JANET: "Okay, I've set the volume to 55%."     <- pactl still said 80%

Nothing ran. The rescue path sends an utterance to GENERAL, GENERAL performs no
action, and the model — asked to reply helpfully to "turn it down" with nothing
to report — narrated the action instead of doing it. Its stated reasoning even
said "the system fact confirms that the volume is now 55%", when FACTS was None.

For an assistant with no screen, "silently didn't do it, but said it did" is
worse than any error message: you walk away believing the light is off.

The rule this module enforces is narrow and checkable: **FACTS are the only
evidence an action ran.** When a reply claims a completed action and there are
no facts behind it, the claim is false by construction, whatever the sentence
says. The prompt asks the model not to do this; this is the part that does not
depend on the model complying, because prompting has already been measured
failing at exactly this kind of instruction elsewhere in JANET (see selfmod).
"""
import re

# Verbs that assert JANET changed the world. Deliberately NOT things like
# "checked", "looked" or "found" — reporting is fine without facts, acting isn't.
_ACTION_VERBS = (
    r"set|added|add|created|create|turned|turn|switched|switch|sent|send|"
    r"deleted|delete|removed|remove|scheduled|schedule|cancelled|canceled|"
    r"cancel|changed|change|updated|update|adjusted|adjust|muted|mute|"
    r"unmuted|started|start|stopped|stop|dimmed|dim|locked|unlocked|saved|save"
)

# First person, completed: "I've set", "I have added", "I just turned".
_FIRST_PERSON = re.compile(
    rf"\bi(?:'ve|\s+have|\s+did)?\s+(?:just\s+|now\s+|already\s+|gone\s+ahead\s+and\s+)?"
    rf"(?:{_ACTION_VERBS})\b",
    re.IGNORECASE,
)

# Fixed phrases that mean the same thing without naming a verb. Anchored to the
# start of a sentence so "when you're done" and "I'm done talking" don't match.
_DONE = re.compile(
    r"(?:^|[.!?]\s+)(?:done|all set|consider it done|that'?s done|it'?s done)\b",
    re.IGNORECASE,
)

# Asserting the NEW STATE instead of the action. Found immediately after the
# first-person rule went in: told not to say "I've set the volume", the model
# said "Okay, the volume is now at 55%" instead — same lie, different grammar.
# Reporting a state you did not read is exactly as false as claiming the act.
#
# Bare "is on"/"is off" is deliberately excluded ("Everest is on the border of
# Nepal") — those only count next to "now", "turned/switched", or a device.
_STATE_CHANGED = re.compile(
    r"\bnow\s+(?:at|on|off|set|muted|unmuted)\b"
    r"|\b(?:is|are|it'?s|that'?s)\s+now\b"
    r"|\b(?:is|are|it'?s|that'?s)\s+(?:set|scheduled|cancelled|canceled|added|"
    r"removed|deleted|updated)\b"
    r"|\b(?:turned|switched)\s+(?:on|off)\b"
    r"|\b(?:volume|lights?|lamp|fan|switch|thermostat|alarm|timer)\b[^.!?]*"
    r"\b(?:is|are)\s+(?:on|off|at)\b",
    re.IGNORECASE,
)

# Said instead of the false claim. Honest about the outcome without pretending
# to know why, because at this point we genuinely don't.
REFUSAL = "Sorry, I wasn't actually able to do that one."


def claims_action(text):
    """Does this reply assert that JANET performed an action, or report a state
    it could only know by having acted?

    Only ever called when FACTS are empty, which is what makes it safe to be
    this aggressive: a genuine state report always arrives as facts from a
    handler, so in the no-facts case there is nothing legitimate left to
    describe. Knowledge answers ("the capital of France is Paris") don't assert
    a changed state and pass through untouched.
    """
    if not text:
        return False
    return bool(_FIRST_PERSON.search(text)
                or _DONE.search(text)
                or _STATE_CHANGED.search(text))
