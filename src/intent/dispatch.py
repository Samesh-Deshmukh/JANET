# src/intent/dispatch.py
from actions import time_action, date_action, timer_action, calc_action
from intent.intent import classify, CONF_THRESHOLD
from intent.normalize import normalize
from intent.scorer import score, THRESHOLD

# The classifier's confidence adds a small bonus to the addressing score:
# bonus = round(confidence * CONF_BONUS_SCALE). At 10 a top-confidence intent is
# worth ~10 points — a tiebreaker that rescues borderline scores without letting
# a confident classification override the linguistic signal (which is what tells
# a command TO JANET apart from a statement ABOUT the same topic).
CONF_BONUS_SCALE = 10

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

    Layer 2 runs first here (we need its confidence to boost the addressing
    score): the classifier must return a real intent it's sure of, AND the
    confidence-boosted linguistic score must clear the threshold. Either veto
    => None (silent).
    """
    # Normalize once so both layers see the clean form the dataset used
    # ("What's the time?" -> "whats the time"); Whisper's caps/punctuation would
    # otherwise make the scorer miss every signal and skew the classifier.
    query = normalize(query)

    # Layer 2: what is it, and is the classifier sure?
    label, confidence, slots = classify(query)
    vetoed = label == "NONE" or confidence < CONF_THRESHOLD
    print(f"🧠 Intent: {label} ({confidence:.0%}){' — ignored' if vetoed else ''}")
    if vetoed:
        return None                              # NONE / low confidence -> stay silent

    # Layer 1: was it addressed to JANET? Linguistic signals + a small confidence bonus.
    ling, breakdown = score(query)
    bonus = round(confidence * CONF_BONUS_SCALE)
    combined = ling + bonus
    detail = ", ".join(f"{name} +{pts}" for name, pts in breakdown) or "no signals"
    addressed = combined >= THRESHOLD
    print(f"🛡  Score: {ling}+{bonus}={combined} ({detail}; conf +{bonus}) "
          f"→ {'addressed' if addressed else 'ignored'}")
    if not addressed:
        return None                              # not addressed -> stay silent

    reply = dispatch(label, slots, ctx)
    if reply is None:
        return "I can't help with that yet."     # addressed + known intent, no handler built
    return reply
