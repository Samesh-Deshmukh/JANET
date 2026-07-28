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

from ai_core import llm, transcript

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
    "\n"
    "Also return:\n"
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

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "reasoning": {"type": "string"},
        "need_deeper_thinking": {"type": "boolean"},
        "interim": {"type": "string"},
    },
    "required": ["reply", "reasoning", "need_deeper_thinking", "interim"],
}

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
    turn += f"USER: {query}"
    messages.append({"role": "user", "content": turn})
    return messages


def _fallback(facts, reason):
    """No model: speak the handler's own words, and say why once."""
    global _announced_outage
    text = facts or NO_FACTS_FALLBACK
    if not _announced_outage:
        _announced_outage = True
        text = f"{OUTAGE_NOTE} {text}"
    print(f"⚠  LLM unavailable ({reason}) — falling back to canned reply")
    return Reply(text=text, reasoning="llm unavailable", facts=facts or "",
                 used_fallback=True)


def compose(query, intent=None, facts=None, history=None, speak=None):
    """Turn facts + context into what JANET says. Never raises for an outage."""
    global _announced_outage
    started = time.time()
    messages = _build_messages(query, intent, facts, history)

    try:
        data = llm.chat(messages, schema=RESPONSE_SCHEMA)
    except llm.LLMUnavailable as exc:
        reply = _fallback(facts, exc)
        _record(query, intent, facts, reply, started)
        return reply

    _announced_outage = False        # the model is back
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
        reply = _fallback(facts, "model returned an empty reply")

    _record(query, intent, facts, reply, started)
    return reply


def _record(query, intent, facts, reply, started):
    transcript.log(
        query=query,
        intent=intent,
        facts=facts,
        reply=reply.text,
        reasoning=reply.reasoning,
        thinking=reply.thinking,
        used_fallback=reply.used_fallback,
        latency_s=round(time.time() - started, 2),
        model=llm._model(),
    )
