# src/intent/dispatch.py
from actions import time_action, date_action, timer_action, calc_action
from intent.intent import decide_action
from intent.scorer import score, THRESHOLD

REGISTRY = {
    "TIME": time_action.handle,
    "DATE": date_action.handle,
    "TIMER": timer_action.handle,
    "CALC": calc_action.handle,
}


def dispatch(intent, slots, ctx):
    """Look up and run the handler. Returns None when nothing is registered.
    Handler exceptions are intentionally NOT caught — bugs stay visible."""
    handler = REGISTRY.get(intent)
    if handler is None:
        return None
    return handler(slots, ctx)


def respond(query, ctx):
    """Full pipeline through JANET's two veto layers.

    Returns the reply text to speak, or None to STAY SILENT — silence is the
    correct output when the utterance wasn't for JANET, so we no longer say a
    fallback line at every overheard sentence.

    Layer 1 (scorer): was this addressed to JANET at all?
    Layer 2 (classifier, inside decide_action): a real intent, or NONE/unsure?
    Either layer vetoing => None (silent).
    """
    total, breakdown = score(query)
    detail = ", ".join(f"{name} +{pts}" for name, pts in breakdown) or "no signals"
    addressed = total >= THRESHOLD
    print(f"🛡  Score: {total} ({detail}) → {'addressed' if addressed else 'ignored'}")
    if not addressed:
        return None                              # Layer 1 veto: not for JANET

    intent, slots = decide_action(query)         # prints the classifier's 🧠 line
    if intent is None:
        return None                              # Layer 2 veto: NONE / low confidence

    reply = dispatch(intent, slots, ctx)
    if reply is None:
        return "I can't help with that yet."     # addressed + known intent, no handler built
    return reply
