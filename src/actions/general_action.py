# src/actions/general_action.py
"""GENERAL intent handler — open-ended questions, answered by the LLM itself.

This file used to own an `ollama.chat` call. It doesn't any more: since
`ai_core/responder.py` composes *every* spoken reply, doing it here too would be
a second, redundant round trip to the same model.

GENERAL is simply the intent with **no facts to report**. There is no action to
run and nothing to look up — the answer comes from the model's own knowledge. So
this handler returns `None`, which tells the responder "no FACTS, just answer
the question."

Kept as its own file rather than deleted so `dispatch.REGISTRY` still reads as
one handler per intent, and so there's an obvious home for future behaviour
(citing sources, refusing certain topics, a web-search tool in slice 2).
"""


def handle(slots, ctx):
    """No facts — the responder answers from the model's own knowledge."""
    return None
