# src/actions/smart_home_action.py
"""SMART_HOME intent handler — switch lights/switches/fans on and off, and answer
"is the X on?".

The classifier only tells us the intent is SMART_HOME; *which* device and *what
to do* live in the words, so this file reads the raw transcript from `ctx.query`
(no slot needed in intent.py) and parses it here.

SAFETY — deliberate exclusion: on/off (and state questions) for lights, switches
and fans only. No locks, garage doors, covers, or alarm panels. JANET has no wake
word, so a security-relevant action must never hang off one misheard sentence;
the backend filters those domains out too (see integrations/homeassistant_source.py).
"""
from collections import Counter

from integrations.smart_home_factory import get_smart_home_source
from integrations.smart_home_source import SmartHomeUnavailable
from intent.normalize import normalize

NOT_SET_UP = "Smart home isn't set up yet."
UNREACHABLE = "I couldn't reach your smart home."
NOT_FOUND = "I couldn't find a device called that."
UNCLEAR = "I'm not sure what you want me to switch on or off."

# "turn on the fan" / "switch off the lights" — verb and particle side by side.
_PHRASES = [
    ("turn on", "on"), ("switch on", "on"), ("put on", "on"),
    ("turn off", "off"), ("switch off", "off"), ("put off", "off"),
]
# "turn the fan on" — same verbs, particle pushed to the end of the sentence.
_VERBS = ("turn", "switch", "put")

# Words that carry no device name. Dropped so "the living room lights please"
# and "living room lights" match the same device.
_FILLER = {"the", "a", "an", "my", "our", "please", "janet", "can", "you",
           "could", "hey", "ok", "okay", "some", "all"}


def _clean_target(text):
    """The device words left after the filler is removed."""
    return " ".join(w for w in text.split() if w not in _FILLER)


def parse_command(text):
    """(action, target) from a NORMALIZED query.

    action is "on", "off", "state" (a question about a device), or None when the
    sentence isn't a switching command at all.
    """
    words = text.split()

    # "turn on the kitchen lights"
    for phrase, action in _PHRASES:
        if phrase in text:
            return action, _clean_target(text.replace(phrase, " "))

    # "turn the kitchen lights on" — verb first, particle last.
    if len(words) > 2 and words[0] in _VERBS and words[-1] in ("on", "off"):
        return words[-1], _clean_target(" ".join(words[1:-1]))

    # "is the fan on" / "are the kitchen lights off"
    if len(words) > 2 and words[0] in ("is", "are") and words[-1] in ("on", "off"):
        return "state", _clean_target(" ".join(words[1:-1]))

    return None, ""


def _word_weights(devices):
    """How much each word narrows down *which* device you meant.

    "lights" appears in several device names, so hearing it barely helps;
    "bedroom" appears in one, so it's decisive. Weight = 1 / (number of device
    names containing the word), which makes rare words worth more. Without this,
    "the bedroom lights" would match "Living Room Lights" (shares "lights") just
    as well as "Bedroom Lamp" (shares "bedroom").
    """
    counts = Counter(word for d in devices for word in set(normalize(d.name).split()))
    return {word: 1 / count for word, count in counts.items()}


def match_device(target, devices):
    """The device the spoken `target` refers to, or None.

    Deliberately simple and explainable, no fuzzy-matching library: an exact
    (normalized) name wins outright; otherwise the device whose shared words
    are worth the most (see `_word_weights`). At least one shared word is
    required, so an unknown device returns None instead of a random light.
    """
    if not target:
        return None
    target_words = set(target.split())
    weights = _word_weights(devices)
    best, best_score = None, 0
    for device in devices:
        name = normalize(device.name)
        if name == target:
            return device
        score = sum(weights[w] for w in target_words & set(name.split()))
        if score > best_score:          # strict >, so ties keep the first device
            best, best_score = device, score
    return best


def _state_sentence(device):
    """'The living room lights are on.' / 'The fan is off.' Crude but reliable
    plural rule: a name ending in "s" takes "are"."""
    name = device.name.lower()
    verb = "are" if name.endswith("s") else "is"
    return f"The {name} {verb} {device.state}."


# The backend is built once per process and reused. The demo backend keeps its
# on/off state in memory, so building a fresh one each turn would forget the
# light you just switched on. `None` is a real answer ("not configured"), hence
# the sentinel rather than `if _source is None`.
_UNSET = object()
_source = _UNSET


def _get_source():
    global _source
    if _source is _UNSET:
        _source = get_smart_home_source()
    return _source


def handle(slots, ctx):
    """SMART_HOME entry point. `slots` is unused — the device name and the verb
    both come from the raw transcript on ctx.query."""
    source = _get_source()
    if source is None:
        return NOT_SET_UP

    action, target = parse_command(normalize(ctx.query))
    if action is None:
        return UNCLEAR

    try:
        devices = source.list_devices()
    except SmartHomeUnavailable:
        return UNREACHABLE

    device = match_device(target, devices)
    if device is None:
        return NOT_FOUND

    if action == "state":
        return _state_sentence(device)

    try:
        if action == "on":
            source.turn_on(device)
        else:
            source.turn_off(device)
    except SmartHomeUnavailable:
        return UNREACHABLE
    return f"Turned {action} the {device.name.lower()}."
