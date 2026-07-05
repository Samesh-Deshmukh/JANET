# src/intent/dispatch.py
from actions import time_action, date_action, timer_action, calc_action
from intent.intent import decide_action

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
    """Full pipeline: query -> intent -> handler text (or fallback line)."""
    intent, slots = decide_action(query)
    reply = dispatch(intent, slots, ctx)
    if reply is None:
        return "Sorry, I didn't catch that."
    return reply
