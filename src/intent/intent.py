import re

from intent import classifier

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}

# Below this softmax probability the classifier isn't sure enough to act — treated
# the same as NONE (a veto). Above it, the confidence also feeds a small bonus to
# the addressing score (see dispatch.respond).
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


def classify(query):
    """Classify one (normalized) utterance.

    Returns (label, confidence, slots). The caller decides whether to act — this
    just reports what the classifier saw and the slots that label would need.
    """
    label, confidence = classifier.predict(query)
    return label, confidence, _slots_for(label, query)
