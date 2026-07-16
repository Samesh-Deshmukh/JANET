# src/actions/general_action.py
"""GENERAL intent handler — answers open-ended questions with a local LLM.

The classifier routes anything that isn't a structured skill (TIME, TIMER, ...)
here. We hand the raw question to a local Ollama model and speak its reply. This
is the only handler that talks to an LLM; everything stays on-device.
"""
import ollama

# Single knob: swap to qwen3:8b / gemma3:12b / gemma4 in one line to A/B.
MODEL = "qwen3:14b"

# The most important line in this file. Answers are READ ALOUD, so they must be
# short and plain — LLMs ramble and emit markdown by default, both unbearable
# over voice.
SYSTEM_PROMPT = (
    "You are JANET, a local voice assistant. Answer in one or two short, "
    "spoken sentences. Be direct and conversational. Never use markdown, "
    "lists, code blocks, or emoji — your reply is read aloud."
)

# Spoken when the Ollama server is down or the model is missing. This is an
# operational failure, not a code bug, so we speak a graceful line instead of
# crashing the assistant.
UNAVAILABLE = "Sorry, my language model isn't available right now."


def handle(slots, ctx):
    """Answer a general question via the local LLM. `slots` is unused (GENERAL
    carries no slots); the question is the raw transcript on ctx.query."""
    question = ctx.query.strip()
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            think=False,   # qwen3 can "think"; we don't want it spoken or slow
        )
    except (ConnectionError, ollama.ResponseError):
        # server unreachable / model not pulled — operational, not a bug
        return UNAVAILABLE
    return resp["message"]["content"].strip()
