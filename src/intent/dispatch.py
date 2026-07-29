# src/intent/dispatch.py
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


def _rescue(raw_query, ling, ctx, why):
    """Before going silent, ask the LLM whether it was being talked to.

    Both gates judge one sentence alone, so a follow-up like "and tomorrow?" is
    invisible to them — the LLM is the only part of JANET holding the
    conversation. A rescued utterance goes to GENERAL, which lets the responder
    answer it and call whatever tool it needs.
    """
    if not addressing.should_check(ling, ctx.history, THRESHOLD - CONF_BONUS_SCALE):
        print(f"🛡  {why} → ignored")
        return None
    addressed, reason = addressing.is_addressed(raw_query, ctx.history)
    if not addressed:
        print(f"🤔 {why} → not for me ({reason})")
        return None
    print(f"🤔 {why} → rescued: {reason}")
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
        return _rescue(raw_query, ling, ctx, f"Intent: {label} ({confidence:.0%})")
    print(f"🧠 Intent: {label} ({confidence:.0%})")

    # The confidence bonus is only needed when the linguistic score fell short --
    # a passing score doesn't change by adding to it.
    bonus = round(confidence * CONF_BONUS_SCALE) if ling < THRESHOLD else 0
    combined = ling + bonus
    shown = f"{ling}+{bonus}={combined}" if bonus else f"{ling}"
    addressed = combined >= THRESHOLD
    print(f"🛡  Score: {shown} ({detail}) → {'addressed' if addressed else 'ignored'}")
    if not addressed:
        return _rescue(raw_query, ling, ctx, f"Score: {shown} borderline")

    # The handler no longer speaks: whatever it returns is the FACTS, and the
    # responder turns those facts + the conversation into what JANET says.
    # GENERAL returns None on purpose — nothing to report, the LLM just answers.
    facts = dispatch(label, slots, ctx)
    # score/confidence don't change the reply — they go into the transcript so the
    # log shows WHY this utterance got through both gates.
    return responder.compose(raw_query, label, facts, ctx.history, ctx.speak,
                             score=combined, confidence=confidence)
