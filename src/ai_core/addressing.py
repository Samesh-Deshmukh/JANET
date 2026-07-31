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

# Inside the follow-up window, only SHORT utterances are worth a model call.
# See should_check for the measurement behind it.
WINDOW_MAX_WORDS = int(os.environ.get("JANET_FOLLOWUP_MAX_WORDS", "12"))

# The two things a rescued utterance can be. Named so `dispatch` and this module
# can't drift apart on a string literal.
NEW_REQUEST = "new_request"
FOLLOW_UP = "follow_up"

_SCHEMA = {
    "type": "object",
    "properties": {
        "addressed": {"type": "boolean"},
        # Which KIND of thing it is decides where a rescued utterance goes, and
        # getting this wrong breaks one case or the other:
        #   "new_request" -> route to the real handler. "Can you turn it up to
        #     60" classified SYSTEM at 36%, under the confidence floor, and was
        #     sent to GENERAL — which has no hands — so JANET said "I can't do
        #     that yet" about something it does perfectly well.
        #   "follow_up"   -> GENERAL, so the responder's tools resolve it from
        #     context. "and tomorrow?" classifies CALENDAR at 38%; routing it to
        #     the calendar handler would read out events instead of tomorrow's
        #     weather. This is the case the rescue was BUILT for.
        "kind": {"type": "string", "enum": ["new_request", "follow_up"]},
        "reason": {"type": "string"},
    },
    "required": ["addressed", "kind", "reason"],
}

# Deliberately narrow. This model is NOT deciding what to do or how to answer —
# only whether the words were aimed at it. Saying that plainly keeps it from
# drifting into being helpful about the content.
PROMPT = (
    "You are JANET, a voice assistant in someone's home. Your microphone is "
    "always on, so most of what you hear is NOT for you: people talking to each "
    "other, a television, a video, a podcast.\n"
    "\n"
    "Decide ONE thing about the new utterance: was it aimed at YOU?\n"
    "\n"
    "YES if any of these is true:\n"
    "  1. you asked a question and this could be the answer — however short or "
    "vague (\"oh yeah\", \"8 am every weekday\", \"no it's okay\");\n"
    "  2. it only makes sense as a continuation of what you just said "
    "(\"and tomorrow?\", \"why?\", \"what about Delhi\");\n"
    "  3. it asks you to do or tell them something you could actually do — a "
    "calculation, the time, a timer, a light. A change of subject is fine.\n"
    "\n"
    "NO if any of these is true:\n"
    "  4. it is part of a longer explanation, story or argument — narration, "
    "not a request. Media sounds like this;\n"
    "  5. it names or addresses someone else, or is two people talking;\n"
    "  6. it only closes the exchange — \"okay\", \"thanks\", \"got it\";\n"
    "  7. it asks for something physical. You have no body: you cannot pass, "
    "fetch, carry or open anything.\n"
    "\n"
    "A question is not enough on its own. Media asks questions constantly — of "
    "the viewer, of another character, rhetorically. Ask who would answer it.\n"
    "\n"
    "When you cannot tell, answer NO. Staying quiet is the safe mistake.\n"
    "\n"
    "Your speech-to-text is unreliable and mangles your own name (\"Janet\" "
    "becomes \"In January\", \"Jan at\", \"Janette\"), so never require the "
    "name to be present or spelled correctly.\n"
    "\n"
    "`reason`: quote the words that decided it and say what they show. Never "
    "restate a rule from this list — \"it follows on from what you just said\" "
    "explains nothing.\n"
    "\n"
    "`kind`: \"new_request\" if you would DO something about it; \"follow_up\" "
    "if answering means using what was just said. When both fit, choose "
    "\"new_request\" — refusing something you can do is worse than acting on "
    "the wrong thing."
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
    # We asked a question, so we are OWED an answer: check regardless of score,
    # and regardless of how long they took to think about it. Every one of these
    # was missed in live testing because the answer to "what time should I set
    # the alarm for?" scores 0 like any other ambient sentence.
    # A reply to JANET, or an answer to its question, is SHORT. The longest
    # genuine follow-up measured across a real session was 8 words; the media
    # chunk that leaked ("...can you think of the smallest butt head double? I'll
    # give you a hint...") was 31. Length is free to check and separates them
    # cleanly: at this limit 100% of real follow-ups survive and 46% of ambient
    # stops costing a model call at all.
    if len(query.split()) > WINDOW_MAX_WORDS:
        return score >= floor
    if history is not None and history.last_reply_was_question():
        return True
    since = history.seconds_since_last() if history is not None else None
    if since is not None and since <= WINDOW_SECONDS:
        return True
    return score >= floor


def is_addressed(query, history):
    """(addressed, kind, reason). Never raises — on any LLM problem, returns False.

    `kind` is "new_request" or "follow_up" and decides where a rescued utterance
    is routed — see `_SCHEMA`. False is the safe default: it just means JANET
    stays quiet, which is what would have happened anyway without this module.
    """
    messages = [{"role": "system", "content": PROMPT}]
    if history is not None:
        messages += history.messages()
        block = history.context_block()
        if block:
            messages.append({"role": "system", "content": block})
        # State it outright rather than relying on the model noticing the "?" at
        # the end of its own last message. The same lesson as FACTS in the
        # responder: an explicit sentence is obeyed, an absence is interpreted.
        if history.last_reply_was_question():
            messages.append({"role": "system", "content":
                             "NOTE: your last message was a question, and it "
                             "has not been answered yet. What you just heard is "
                             "very likely that answer."})
    messages.append({"role": "user", "content":
                     f'The person just said: "{query}". Are they talking to you?'})
    try:
        data = llm.chat(messages, schema=_SCHEMA, max_tokens=160)
    except llm.LLMUnavailable as exc:
        print(f"⚠  addressing check unavailable ({exc}) — staying silent")
        return False, FOLLOW_UP, "llm unavailable"
    kind = (data.get("kind") or FOLLOW_UP).strip()
    if kind not in (NEW_REQUEST, FOLLOW_UP):
        kind = FOLLOW_UP                 # unknown value: the safer of the two
    return bool(data.get("addressed")), kind, (data.get("reason") or "").strip()
