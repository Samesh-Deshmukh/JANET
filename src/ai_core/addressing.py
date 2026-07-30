# src/ai_core/addressing.py
"""Last chance before silence: ask the LLM whether it was being talked to.

JANET's two gates — the [scorer] and the [classifier] — both judge ONE sentence
in isolation. Neither can see what JANET just said, and that is exactly what a
follow-up depends on:

    you:   Janet, what's the weather?
    JANET: It's 22 degrees and partly cloudy in Pune.
    you:   and tomorrow?

"and tomorrow" has no "janet", no question verb and no keyword, so the scorer
gives it **0** and it never reaches the classifier. Question-shaped follow-ups
fail the other way: "how about friday" passes the scorer with 40, then the
classifier — seeing four contextless words — manages only 38% on CALENDAR and
vetoes it. Measured, both of them.

The LLM is the only part of JANET that actually *has* the conversation, so it is
the only thing that can answer "was that meant for me?". This module asks it,
but only when it's worth asking (see `should_check`), because the whole point of
the cheap gates is not paying a model for every overheard sentence.

A rescued utterance is routed to GENERAL, so the responder answers it and can
call whatever tool it needs — "and tomorrow?" becomes a `get_weather` lookup.
"""
import os

from ai_core import llm

# How recently JANET must have spoken for an utterance to count as "probably a
# follow-up". Long enough for a real reply, short enough that the room goes back
# to being ignored quickly.
WINDOW_SECONDS = float(os.environ.get("JANET_FOLLOWUP_WINDOW", "30"))

# Nobody speaks a 40-word command to an assistant. Continuous speech — a video, a
# podcast, two people talking — never gives the VAD enough silence to end an
# utterance, so it grows until main.py's 30-second cap cuts it. Measured live: a
# Python tutorial produced repeated 30s chunks of ~80-100 words, and by sheer
# length they accumulated enough incidental question words and keywords to clear
# the score floor, reach the classifier, and buy an LLM addressing check.
#
# So the "background TV costs nothing" property held only for SHORT ambient
# fragments, and was luck-of-the-draw for long ones. Length is the cheapest
# possible signal that something wasn't addressed to JANET, and it needs no
# plumbing — the transcript is already here.
MAX_WORDS = int(os.environ.get("JANET_MAX_COMMAND_WORDS", "40"))

_SCHEMA = {
    "type": "object",
    "properties": {
        "addressed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["addressed", "reason"],
}

# Deliberately narrow. This model is NOT deciding what to do or how to answer —
# only whether the words were aimed at it. Saying that plainly keeps it from
# drifting into being helpful about the content.
PROMPT = (
    "You are JANET, a local voice assistant in someone's home. Your microphone "
    "is always on, so you constantly hear speech that is NOT for you: other "
    "people talking to each other, a TV, a video playing.\n"
    "\n"
    "Below is your recent conversation, including what your actions returned. "
    "Decide ONE thing about the new utterance: is the person speaking to YOU?\n"
    "\n"
    "Say YES in either of these cases:\n"
    "  * it follows on from what you just said — a short question or reply that "
    "only makes sense as a continuation (\"and tomorrow?\", \"why?\", \"do "
    "that\", \"what about Delhi\");\n"
    "  * it is a request of the kind people make OF AN ASSISTANT — a "
    "calculation, a fact, the time, a timer, a command — even if it changes the "
    "subject completely. Changing topic does not mean they stopped talking to "
    "you.\n"
    "\n"
    "Say NO when it is people talking to each other, background media, or a "
    "statement nobody expects you to act on. When you genuinely can't tell, say "
    "no: staying quiet is the safe mistake.\n"
    "\n"
    "Note your speech-to-text is unreliable and often mangles your own name — "
    "\"Janet\" comes through as \"In January\", \"Jan at\", \"Janette\". "
    "Do not require your name to be spelled correctly, or present at all."
)


def should_check(score, history, floor, query=""):
    """Is it worth paying for the LLM on this one?

    Two triggers, either is enough:

    * **JANET spoke within WINDOW_SECONDS.** This is the one that matters — a
      follow-up like "and tomorrow?" scores 0 and is invisible to the gates, but
      it arrives seconds after a reply.
    * **The score reached `floor`** (dispatch's `THRESHOLD - CONF_BONUS_SCALE`).
      That's the band the pipeline already treats as "plausibly addressed", so
      an utterance that got there and still failed is worth a second look.

    `floor` is deliberately NOT "any signal above zero". Ambient speech picks up
    a stray keyword constantly — "he set an alarm yesterday" scores 25 — and
    checking those would mean an LLM call for half the sentences in a room with
    a TV on, in exchange for almost nothing. Below the floor, the pipeline's own
    design already says the utterance can't be rescued on linguistic grounds, so
    only a recent reply justifies asking.
    """
    # Far too long to be something said TO an assistant — almost certainly media
    # or a conversation. Checked before anything else because it is free.
    if len(query.split()) > MAX_WORDS:
        return False
    since = history.seconds_since_last() if history is not None else None
    if since is not None and since <= WINDOW_SECONDS:
        return True
    return score >= floor


def is_addressed(query, history):
    """(addressed, reason). Never raises — on any LLM problem, returns False.

    False is the safe default: it just means JANET stays quiet, which is what
    would have happened anyway without this module.
    """
    messages = [{"role": "system", "content": PROMPT}]
    if history is not None:
        messages += history.messages()
        block = history.context_block()
        if block:
            messages.append({"role": "system", "content": block})
    messages.append({"role": "user", "content":
                     f'The person just said: "{query}". Are they talking to you?'})
    try:
        data = llm.chat(messages, schema=_SCHEMA, max_tokens=120)
    except llm.LLMUnavailable as exc:
        print(f"⚠  addressing check unavailable ({exc}) — staying silent")
        return False, "llm unavailable"
    return bool(data.get("addressed")), (data.get("reason") or "").strip()
