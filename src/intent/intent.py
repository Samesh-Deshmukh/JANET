import re

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}
_CALC_WORDS = (" plus ", " minus ", " times ", " divided by ", " multiplied by ")


def _parse_duration(query):
    """Pull '<n> second|minute|hour' out of the query; default 5 minutes."""
    match = re.search(r"(\d+)\s*(second|minute|hour)", query)
    if not match:
        return 300
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def decide_action(query):
    q = query.lower()
    if "time" in q and ("what" in q or "tell" in q):
        return ("TIME", {})
    if "date" in q or "what day" in q:
        return ("DATE", {})
    if "timer" in q:
        return ("TIMER", {"duration": _parse_duration(q)})
    if "calculate" in q or any(word in q for word in _CALC_WORDS):
        return ("CALC", {"expression": q})
    return (None, {})
