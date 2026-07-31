# src/intent/dispatch.py
import re

from actions import (
    time_action, date_action, timer_action, calc_action, general_action,
    alarm_action, calendar_action, weather_action, smart_home_action,
    system_action, reminder_action, email_action,
)
from intent.intent import classify, CONF_THRESHOLD
from intent.normalize import normalize
from intent.scorer import score, THRESHOLD
from ai_core import addressing, responder
from utils import confirm

# The classifier's confidence adds a small bonus to the addressing score:
# bonus = round(confidence * CONF_BONUS_SCALE). At 10 a top-confidence intent is
# worth ~10 points — a tiebreaker that rescues borderline scores without letting
# a confident classification override the linguistic signal (which is what tells
# a command TO JANET apart from a statement ABOUT the same topic).
CONF_BONUS_SCALE = 10

# "janet", "hey janet", "janet uh..." — the name and nothing else. Matched
# against the NORMALIZED text, which has already lost its punctuation.
_NAME_ONLY = re.compile(r"(?:hey\s+|hi\s+|ok\s+|okay\s+)?janet(?:\s+(?:uh|um|er|erm|hmm|so|well))*")

REGISTRY = {
    "TIME": time_action.handle,
    "DATE": date_action.handle,
    "TIMER": timer_action.handle,
    "CALC": calc_action.handle,
    "GENERAL": general_action.handle,
    "ALARM": alarm_action.handle,
    "CALENDAR": calendar_action.handle,
    "WEATHER": weather_action.handle,
    "SMART_HOME": smart_home_action.handle,
    "SYSTEM": system_action.handle,
    "REMINDER": reminder_action.handle,
    "EMAIL": email_action.handle,
}


def dispatch(intent, slots, ctx):
    """Look up and run the handler. Returns None when nothing is registered.
    Handler exceptions are intentionally NOT caught — bugs stay visible."""
    handler = REGISTRY.get(intent)
    if handler is None:
        return None
    return handler(slots, ctx)


# Signals that are real evidence the utterance was aimed at JANET. "question"
# and "2nd-person" are NOT here: media is full of both.
_STRONG_SIGNALS = {"janet", "keyword", "command"}


def _weakly_addressed(label, breakdown):
    """Passed the gates, but with no evidence beyond being question-shaped."""
    return (label == "GENERAL"
            and not ({name for name, _ in breakdown} & _STRONG_SIGNALS))


def _rescue(raw_query, ling, ctx, why, label=None, slots=None):
    """Before going silent, ask the LLM whether it was being talked to.

    Both gates judge one sentence alone, so a follow-up like "and tomorrow?" is
    invisible to them — the LLM is the only part of JANET holding the
    conversation.

    Where a rescued utterance GOES depends on what kind of thing it is, and
    getting that wrong breaks one case or the other. Live testing found both:

    * "Can you turn it up to 60?" classified SYSTEM at **36%**, under the
      confidence floor, and went to GENERAL — which runs no action — so JANET
      answered "I can't do that yet" about something it does perfectly well.
      Three different capabilities were falsely refused this way in one session.
    * "and tomorrow?" classifies CALENDAR at **38%**. Routing *that* to the
      calendar handler would read out events instead of tomorrow's weather.

    So the addressing check now also reports `kind`, and a **new_request** is
    sent to the classifier's handler while a **follow_up** goes to GENERAL,
    where the responder's tools can resolve it from context.

    The confidence floor is deliberately not consulted here. It was already
    applied — and overruled — by the addressing check, which had the whole
    conversation to look at and said this really was a request. Re-applying it
    would just reproduce the bug.
    """
    if not addressing.should_check(ling, ctx.history, THRESHOLD - CONF_BONUS_SCALE,
                                   raw_query):
        print(f"🛡  {why} → ignored")
        return None
    addressed, kind, reason = addressing.is_addressed(raw_query, ctx.history)
    if not addressed:
        print(f"🤔 {why} → not for me ({reason})")
        return None
    print(f"🤔 {why} → rescued ({kind}): {reason}")

    # The scorer can veto before the classifier ever runs, so a rescued
    # utterance may not have a label yet. Classify it now rather than earlier:
    # ambient speech never reaches this line, so the common case still pays
    # nothing (and warm, the classifier costs ~0ms anyway).
    if kind == addressing.NEW_REQUEST and label is None:
        label, _confidence, slots = classify(normalize(raw_query))

    if kind == addressing.NEW_REQUEST and label and label in REGISTRY:
        print(f"↪  routing to {label}")
        facts = dispatch(label, slots, ctx)
        return responder.compose(raw_query, label, facts, ctx.history, ctx.speak,
                                 score=ling)
    return responder.compose(raw_query, "GENERAL", None, ctx.history, ctx.speak,
                             score=ling)


def respond(query, ctx):
    """Full pipeline through JANET's two veto layers.

    Returns an `ai_core.responder.Reply` to speak, or None to STAY SILENT — the
    handler's string is no longer spoken directly; it becomes the FACTS the LLM
    phrases. Silence is still the correct output when the utterance wasn't for
    JANET, so we say nothing at all at an overheard sentence.

    A pending confirmation (utils/confirm) is answered FIRST, before either
    layer — a bare "yes" has no linguistic signal and would be thrown away as
    ambient speech.

    Then Layer 1 (the cheap scorer) runs, and only if the utterance is
    plausibly addressed do we pay for Layer 2: the classifier must return a
    real intent it's sure of, AND the confidence-boosted linguistic score must
    clear the threshold. Either veto => None (silent).
    """
    # Normalize once so both layers see the clean form the dataset used
    # ("What's the time?" -> "whats the time"); Whisper's caps/punctuation would
    # otherwise make the scorer miss every signal and skew the classifier.
    # The RAW form is kept for the responder: the LLM speaks better from natural
    # text than from the stripped, lowercased form the two gates need.
    raw_query = query
    query = normalize(query)

    # A pending confirmation is answered BEFORE the veto layers: a bare "yes"
    # carries no linguistic signal (score 0) and would be ignored as ambient
    # speech. resolve() returns None when the utterance wasn't a yes/no, in which
    # case we fall through and treat it as an ordinary command.
    if confirm.is_pending():
        answer = confirm.resolve(query)
        if answer is not None:
            print(f"✅ Confirmation: {query!r}")
            # Even this goes through the responder, so JANET has ONE voice —
            # the outcome of a confirmed action is spoken like everything else.
            return responder.compose(raw_query, "CONFIRM", answer, ctx.history,
                                     ctx.speak, score=None, confidence=None)

    # Someone saying the name and then trailing off is addressing JANET — they
    # just haven't got to the request yet. Classified, "janet uh" came back
    # SYSTEM at 60% and JANET answered "I'm sorry, I can't do that yet", which
    # is a refusal of a question nobody asked. A prompt is the useful reply.
    if _NAME_ONLY.fullmatch(query):
        print("🛡  Name only → acknowledging")
        return responder.compose(
            raw_query, "SYSTEM",
            "They said your name but haven't asked for anything yet.",
            ctx.history, ctx.speak)

    # Layer 1 (cheap) runs first. If the linguistic score is so low that even a
    # maxed-out confidence bonus couldn't reach the threshold, it can't be
    # rescued -- so we stay silent WITHOUT paying for the classifier. This skips
    # the model on the low-signal ambient chatter that fills a room.
    ling, breakdown = score(query)
    detail = ", ".join(f"{name} +{pts}" for name, pts in breakdown) or "no signals"
    if ling < THRESHOLD - CONF_BONUS_SCALE:
        return _rescue(raw_query, ling, ctx, f"Score: {ling} ({detail})")

    # Plausibly addressed -> we need the classifier anyway (for the intent + the
    # NONE veto), so run it now.
    label, confidence, slots = classify(query)
    if label == "NONE" or confidence < CONF_THRESHOLD:
        # Half of all missed follow-ups die here, not at the scorer: "how about
        # friday" passes the scorer at 40 then gets only 38% on CALENDAR.
        return _rescue(raw_query, ling, ctx, f"Intent: {label} ({confidence:.0%})",
                       label=label, slots=slots)
    print(f"🧠 Intent: {label} ({confidence:.0%})")

    # The confidence bonus is only needed when the linguistic score fell short --
    # a passing score doesn't change by adding to it.
    bonus = round(confidence * CONF_BONUS_SCALE) if ling < THRESHOLD else 0
    combined = ling + bonus
    shown = f"{ling}+{bonus}={combined}" if bonus else f"{ling}"
    addressed = combined >= THRESHOLD
    print(f"🛡  Score: {shown} ({detail}) → {'addressed' if addressed else 'ignored'}")
    if not addressed:
        return _rescue(raw_query, ling, ctx, f"Score: {shown} borderline",
                       label=label, slots=slots)

    # Both gates passed — but on what evidence? A bare question scores exactly
    # THRESHOLD on question-shape alone, so ANY question in the room clears
    # Layer 1, and GENERAL is the classifier's catch-all for "no idea". The
    # conjunction of those two is the exact profile of overheard media: live,
    # "Why are we going this way?" (score 40, GENERAL 66%) was answered.
    #
    # The LLM check exists for precisely this judgement but only ever ran on the
    # SILENT path, so nothing ever second-guessed a wrongly-confident yes.
    # Measured against a real session: this verifies 1 command in 73 (~1s), and
    # catches the one ambient utterance that reached a reply.
    if _weakly_addressed(label, breakdown):
        ok, _kind, reason = addressing.is_addressed(raw_query, ctx.history)
        if not ok:
            print(f"🤔 Verified → not for me ({reason})")
            return None

    # The handler no longer speaks: whatever it returns is the FACTS, and the
    # responder turns those facts + the conversation into what JANET says.
    # GENERAL returns None on purpose — nothing to report, the LLM just answers.
    facts = dispatch(label, slots, ctx)
    # score/confidence don't change the reply — they go into the transcript so the
    # log shows WHY this utterance got through both gates.
    return responder.compose(raw_query, label, facts, ctx.history, ctx.speak,
                             score=combined, confidence=confidence)
