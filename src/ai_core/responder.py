# src/ai_core/responder.py
"""JANET's voice.

Handlers no longer speak. They return FACTS — a short factual string — and this
module turns those facts, plus the conversation so far, into what JANET actually
says. One voice for everything, aware of what was just said.

It also records why it said it (`reasoning`), and can escalate itself to real
step-by-step thinking when a question deserves it.
"""
import os
import time
from dataclasses import dataclass

from ai_core import acting, claims, llm, tools, transcript

# The most important text in the file: these answers are READ ALOUD.
#
# Two rules fight each other and both matter. "Never change a value" keeps JANET
# accurate; "say it the way a person would" keeps it from sounding like a
# database. Benchmarking with only the first rule produced "It is 16:49." and a
# bare "13" — correct and horrible. Hence the explicit examples below.
SYSTEM_PROMPT = (
    "You are JANET, a local voice assistant. You are speaking out loud to the "
    "person who owns you.\n"
    "\n"
    "- Reply in one or two short, natural spoken sentences. Never use markdown, "
    "lists, code blocks, headings or emoji.\n"
    "- FACTS, when present, are the result of an action you just ran. They are "
    "the truth. Never contradict them and never invent a value that isn't "
    "there. If FACTS are missing, answer from your own knowledge.\n"
    "- FACTS are the ONLY evidence that you did something. With no FACTS you "
    "have run nothing and changed nothing, so never say you have set, added, "
    "sent, removed, scheduled or switched anything — you haven't, and they will "
    "believe you. If they asked you to DO something and there are no FACTS yet, "
    "run the matching act_ tool; only if that fails do you say you couldn't. "
    "Answering a question needs no facts; claiming an action always does.\n"
    "- When the FACTS contain a result, that result IS the answer — say it. Only "
    "when the FACTS say the calculation could not be done may you explain a rule "
    "or identity from your own knowledge (sine over cosine is tangent), and even "
    "then do not work out a number yourself.\n"
    "- FACTS describe what JUST happened. Report them as news, not as a state "
    "that was already true: \"Alarm set for tomorrow at 8 AM\" means you have "
    "this moment set it, so say \"I've set an alarm for 8 AM\" — never \"I "
    "already have an alarm set\", which tells them their request did nothing.\n"
    "- Keep every value's meaning exactly, but SAY it the way a person would: "
    "a 24-hour time like 16:49 is \"4:49 PM\"; a bare number like 13 becomes a "
    "sentence such as \"That's 13.\" Never read a value out in a robotic form.\n"
    "- If the FACTS are a QUESTION asking the person to confirm something, your "
    "reply MUST also be a question, ending in a question mark, and must repeat "
    "every specific detail — what will happen, and when. Never turn it into a "
    "statement: \"Shall I add X tomorrow at 4 PM?\" must not become \"I'll add "
    "X tomorrow at 4 PM.\" Nothing has happened yet and they have to be able to "
    "say no.\n"
    "- Use the conversation so far so follow-ups make sense. Don't repeat "
    "yourself word for word if you've just said something similar.\n"
    "- If the FACTS don't cover what was asked, you may LOOK SOMETHING UP: set "
    "tool to the one you need and put its arguments in tool_args. Only do this "
    "when you genuinely lack the information — if the FACTS already answer it, "
    "set tool to 'none' and just reply. When you do look something up, put a "
    "short casual line in interim to say while you fetch it.\n"
    "- If they asked you to DO something and no FACTS show it was done, use the "
    "matching act_ tool instead of apologising. You reach the same machinery the "
    "normal path does, so anything that needs confirming will still ask. Only "
    "act when they actually asked for it — never on an overheard remark, and "
    "never twice for one request.\n"
    "\n"
    "Lookups (these change nothing):\n" + tools.catalogue() + "\n"
    "\n"
    "Actions (these DO something):\n" + acting.catalogue() + "\n"
    "\n"
    "Also return:\n"
    "- reply: the COMPLETE spoken answer. It is the only thing the person "
    "hears, so it must contain the answer itself — if the FACTS name four "
    "events, your reply names them. A reply that promises an answer instead of "
    "giving one is silence, because there is no second turn.\n"
    "- reasoning: one short line on why you replied that way.\n"
    "- need_deeper_thinking: true ONLY when the question genuinely needs "
    "careful multi-step reasoning. Answering from FACTS never does.\n"
    "- interim: a short, casual line to say while you think it over, used only "
    "when need_deeper_thinking is true (e.g. \"hang on, let me think about "
    "that one\")."
)

# The deep-thinking pass runs UNCONSTRAINED (no JSON schema) so the model can
# reason in <think> tags. That means it must NOT be told about the schema fields
# — asked to "also return reasoning", an unconstrained model happily types
# "reasoning: ..." into the answer, and JANET reads it out loud. Caught exactly
# that in testing. So the deep pass gets a speech-only prompt.
DEEP_SYSTEM_PROMPT = (
    "You are JANET, a local voice assistant, speaking out loud. Think the "
    "problem through carefully, then give ONLY your spoken answer: two or three "
    "short natural sentences. No markdown, no lists, no headings, no labels, and "
    "never write out field names like 'reasoning:' — everything you write after "
    "thinking is read aloud verbatim."
)

def schema_for(facts=None):
    """The response schema, offering only the choices this turn can use.

    Actions are offered ONLY when no facts exist yet. If a handler has already
    run and produced facts, the action for this utterance has happened; another
    would be a second action for one request, which is separately refused.

    It also keeps the enum small, which measurably matters. Widening it from 8
    to 15 hurt selection: in 1 run out of 5, "what is 25 percent of 52?" — with
    FACTS of "That's 13." — came back answered with a weather lookup. Every
    extra choice is a chance to pick the wrong one, so nothing is offered that
    this turn has no use for.
    """
    choices = tools.names() + (acting.names() if not facts else [])
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "reasoning": {"type": "string"},
            "need_deeper_thinking": {"type": "boolean"},
            # Lookups and actions share one enum, so constrained decoding makes
            # it IMPOSSIBLE for the model to name either one that doesn't exist
            # — a guarantee native tool-calling APIs don't give you.
            "tool": {"type": "string", "enum": choices},
            "tool_args": {"type": "object"},
            "interim": {"type": "string"},
        },
        "required": ["reply", "reasoning", "need_deeper_thinking", "tool",
                     "tool_args", "interim"],
    }


# The full shape, for callers and tests that just want to inspect it.
RESPONSE_SCHEMA = schema_for()

# How many extra lookups one utterance may trigger. Each costs ~2s, and a model
# that keeps asking for one more would leave the user in silence forever.
MAX_TOOL_ROUNDS = 2

# Deep thinking works (~11s) but JANET is single-threaded, so the mic is deaf for
# that whole window. Off until the owner turns it on deliberately.
DEEP_THINKING = os.environ.get("JANET_DEEP_THINKING", "").lower() in ("1", "true", "yes")

# Said once per outage, so the user knows why replies suddenly got blunt.
OUTAGE_NOTE = "My language model isn't available, so my replies will be basic for now."
NO_FACTS_FALLBACK = "Sorry, I can't help with that right now."

_announced_outage = False        # reset as soon as a call succeeds again


@dataclass
class Reply:
    text: str                    # what JANET says out loud
    reasoning: str = ""          # one line on why
    thinking: str = ""           # the full trace, only when escalated
    facts: str = ""              # what the action returned, kept for history
    used_fallback: bool = False


def reset_outage_notice():
    """Forget that we've announced an outage (used by tests)."""
    global _announced_outage
    _announced_outage = False


def _build_messages(query, intent, facts, history):
    """system + recent chat + this turn's facts and question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history is not None:
        messages += history.messages()
        # What JANET recently did and why — the part plain chat history loses.
        block = history.context_block()
        if block:
            messages.append({"role": "system", "content": block})
    turn = f"INTENT: {intent}\n" if intent else ""
    if facts:
        turn += f"FACTS: {facts}\n"
    else:
        # Say the absence out loud rather than just omitting the line. Silence
        # reads as "no information supplied"; this reads as "you did nothing",
        # which is the fact that matters. With FACTS merely missing, a rescued
        # "turn it down to 55" produced "Okay, the volume is now at 55%" — the
        # model filled the gap with the outcome the user asked for.
        turn += ("FACTS: none — nothing has run YET, so nothing has changed "
                 "yet. If they asked you to DO something, call the matching "
                 "act_ tool now instead of refusing. You may answer a question "
                 "from your own knowledge. Until an action hands back facts, "
                 "never say or imply that anything was done, set, or is now in "
                 "a new state.\n")
    turn += f"USER: {query}"
    messages.append({"role": "user", "content": turn})
    return messages


def _fallback(facts, reason, offline=True):
    """Speak the handler's own words instead of the model's.

    `offline` distinguishes "the server is down" (worth telling the user once,
    because every reply will be blunt until it's back) from "the server answered
    but the answer was empty" — a one-off glitch where claiming an outage would
    be a lie.
    """
    global _announced_outage
    text = facts or NO_FACTS_FALLBACK
    if offline and not _announced_outage:
        _announced_outage = True
        text = f"{OUTAGE_NOTE} {text}"
    print(f"⚠  LLM fallback ({reason}) — using the canned reply")
    return Reply(text=text, reasoning="llm unavailable", facts=facts or "",
                 used_fallback=True)


def compose(query, intent=None, facts=None, history=None, speak=None,
            score=None, confidence=None):
    """Turn facts + context into what JANET says. Never raises for an outage.

    `score`/`confidence` are the two gates' verdicts. They don't affect the reply
    at all — they're threaded through purely so the transcript records WHY this
    utterance was acted on, which is what makes the log usable for tuning the
    scorer and the classifier later.
    """
    global _announced_outage
    started = time.time()
    messages = _build_messages(query, intent, facts, history)

    collected = [facts] if facts else []
    # An action may run at most ONCE per utterance. Measured: asked to add a
    # calendar event, the model called act_calendar on both tool rounds, which
    # queued two confirmations for one request. Harmless there; for act_email
    # that is sending twice. Prompting said "never twice" and did not hold, so
    # this is the part that does not depend on the model complying.
    acted = False
    # Actions are offered only while nothing has run yet — see schema_for().
    turn_schema = schema_for(facts)
    try:
        data = llm.chat(messages, schema=turn_schema)

        # The model may ask to look something up. Run it, hand back the result,
        # and let it answer again — capped so it can't keep stalling.
        for _ in range(MAX_TOOL_ROUNDS):
            wanted = (data.get("tool") or tools.NO_TOOL).strip()
            if wanted == tools.NO_TOOL:
                break
            interim = (data.get("interim") or "").strip()
            if interim and speak:
                speak(interim)       # so the lookup isn't silent
            if wanted in acting.ACTIONS and acted:
                print(f"🚫 Refused a second action ({wanted}) for one request")
                messages.append({"role": "system", "content":
                                 "You have already performed an action for this "
                                 "request. Do not act again — answer now using "
                                 "the FACTS above."})
                data = llm.chat(messages, schema=turn_schema)
                break
            if wanted in acting.ACTIONS:
                acted = True
                # An ACTION, not a lookup. Build the same Context the normal
                # pipeline hands a handler, so confirmation gates and history
                # behave identically whichever way the handler was reached.
                from utils.context import Context
                result = acting.run(wanted, Context(speak=speak or (lambda s: None),
                                                    query=query, history=history))
            else:
                result = tools.run(wanted, data.get("tool_args"))
            if result is None:       # unknown tool: answer with what we have
                break
            print(f"🔧 Tool: {wanted}({data.get('tool_args')}) -> {result}")
            collected.append(result)
            messages.append({"role": "user", "content": f"FACTS: {result}"})
            data = llm.chat(messages, schema=turn_schema)
        else:
            # Ran out of lookups while it still wanted more — make it answer
            # with what it has, or the user just hears another stall line.
            if (data.get("tool") or tools.NO_TOOL) != tools.NO_TOOL:
                messages.append({"role": "system", "content":
                                 "No more lookups are available. Answer now "
                                 "using only the FACTS above."})
                data = llm.chat(messages, schema=turn_schema)
    except llm.LLMUnavailable as exc:
        reply = _fallback(facts, exc)
        _record(query, intent, facts, reply, started, score, confidence)
        return reply

    _announced_outage = False        # the model is back
    facts = "; ".join(collected) if collected else facts
    reply = Reply(
        text=(data.get("reply") or "").strip(),
        reasoning=(data.get("reasoning") or "").strip(),
        facts=facts or "",
    )

    if DEEP_THINKING and data.get("need_deeper_thinking"):
        interim = (data.get("interim") or "").strip()
        if interim and speak:
            speak(interim)          # so the long pause isn't dead air
        try:
            # No schema this time: unconstrained, the model thinks out loud in
            # <think> tags and llm.chat hands them back separately. Swap in the
            # speech-only prompt so schema field names can't leak into speech.
            deep_messages = [{"role": "system", "content": DEEP_SYSTEM_PROMPT}] + messages[1:]
            thinking, answer = llm.chat(
                deep_messages, max_tokens=1200, timeout=llm.DEEP_TIMEOUT_S
            )
            if answer:
                reply = Reply(text=answer, reasoning=reply.reasoning,
                              thinking=thinking, facts=facts or "")
        except llm.LLMUnavailable as exc:
            print(f"⚠  deep thinking failed ({exc}) — keeping the quick answer")

    if not reply.text:
        # The server answered — it just said nothing. Not an outage.
        reply = _fallback(facts, "model returned an empty reply", offline=False)

    # Nothing ran, so nothing can be claimed. The prompt asks for this too, but
    # prompting is not a control — a live test had JANET say "Okay, I've set the
    # volume to 55%" with FACTS of None while pactl still read 80%. See claims.py.
    if not facts and claims.claims_action(reply.text):
        print(f"🚫 Blocked an unbacked action claim: {reply.text!r}")
        reply = Reply(text=claims.REFUSAL,
                      reasoning="claimed an action with no facts behind it",
                      facts="")

    _record(query, intent, facts, reply, started, score, confidence)
    return reply


def _record(query, intent, facts, reply, started, score=None, confidence=None):
    transcript.log(
        query=query,
        intent=intent,
        score=score,
        confidence=confidence,
        facts=facts,
        reply=reply.text,
        reasoning=reply.reasoning,
        thinking=reply.thinking,
        used_fallback=reply.used_fallback,
        latency_s=round(time.time() - started, 2),
        model=llm._model(),
    )
