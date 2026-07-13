import re

from intent import classifier

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}

# Below this softmax probability we don't act on the model's guess. This is a
# placeholder gate: once the multi-signal scorer exists it will own the
# respond/ignore decision and this can relax.
CONF_THRESHOLD = 0.5


def _parse_duration(query):
    """Pull '<n> second|minute|hour' out of the query; default 5 minutes."""
    match = re.search(r"(\d+)\s*(second|minute|hour)", query)
    if not match:
        return 300
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def _slots_for(label, query):
    """Fill the slots a handler needs. The classifier gives the intent TYPE;
    slot extraction stays rule-based (the spec's Layer-2 deterministic parse)."""
    if label == "TIMER":
        return {"duration": _parse_duration(query)}
    if label == "CALC":
        return {"expression": query}
    return {}


def decide_action(query):
    """Classify the utterance, then extract slots for the predicted intent.

    Returns (intent, slots), or (None, {}) when the utterance isn't for JANET
    (NONE) or the model isn't confident enough to act on.
    """
    label, confidence = classifier.predict(query)
    suppressed = label == "NONE" or confidence < CONF_THRESHOLD
    # Show the raw classifier output (incl. NONE / low-confidence cases that
    # main.py's reply line would otherwise hide), and whether we acted on it.
    print(f"🧠 Intent: {label} ({confidence:.0%}){' — ignored' if suppressed else ''}")
    if suppressed:
        return (None, {})
    return (label, _slots_for(label, query))
