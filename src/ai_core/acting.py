# src/ai_core/acting.py
"""The LLM's ability to DO things, not just look them up.

`tools.py` is read-only by design and stays that way — this is the one file that
can change the world, so the write surface is a single place to audit.

## Why this exists

The pipeline used to be the only way to reach a handler: classifier picks a
label, label picks a handler. When the classifier was unsure, the request went
to GENERAL — which has no hands — and JANET said it couldn't do things it does
perfectly well. Measured in one live session:

    "Can you delete lunch with Alex?"  CALENDAR 48%  ->  "I can't delete events"
    "Can you turn it up to 60?"        SYSTEM   36%  ->  "I can't do that yet"

Routing was made more forgiving after that (see `dispatch._rescue`), but that
still leans on a 13-label DistilBERT picking the right label. The model reading
the sentence understands it far better — in every one of those failures its own
rescue reasoning named the right action while the classifier was under its
confidence floor. So it can now name the action directly.

## What this does NOT change

**Every confirmation gate still fires**, because it lives inside the handler,
not in the routing. `act_calendar` calls `calendar_action.handle`, which calls
`confirm.request` and returns a question — so JANET still asks before writing an
event or sending mail, exactly as before. Nothing here bypasses a gate; it only
provides another way to reach the same handler.

Locks, garage doors and alarm panels remain excluded at the smart-home layer,
where they always were.

## The shape

One tool per action rather than a single `act(intent)`, for two reasons that
both came out of earlier testing: the tool name is part of the response schema's
**enum**, so constrained decoding makes it impossible to name an action that
doesn't exist (a free-form `intent` argument would give up that guarantee); and
`description` is load-bearing — with bare names the model once sent "what's on
my calendar tomorrow" to `get_time`.
"""

# name -> (intent label, what it's for). The labels are the same ones the
# classifier produces, so these reach exactly the handlers the normal path does.
#
# TIME, DATE, WEATHER and CALC are deliberately absent: they change nothing, so
# they are lookups and belong in tools.py. GENERAL and NONE are absent because
# neither has an action to run.
ACTIONS = {
    "act_timer": ("TIMER",
                  "set, check or cancel a countdown timer. Use when they ask "
                  "you to time something"),
    "act_alarm": ("ALARM",
                  "set or cancel an alarm at a clock time, including recurring "
                  "ones ('every weekday at 8')"),
    "act_reminder": ("REMINDER",
                     "set, list or cancel a reminder to do something later"),
    "act_calendar": ("CALENDAR",
                     "create or delete a calendar event. Asks the user to "
                     "confirm before it writes anything"),
    "act_smart_home": ("SMART_HOME",
                       "turn a light, switch or fan on or off"),
    "act_system": ("SYSTEM",
                   "change the volume, mute, or repeat what you last said"),
    "act_email": ("EMAIL",
                  "reply to an email. Asks the user to confirm before sending"),
}


def names():
    """Every action name, for the response schema's enum."""
    return list(ACTIONS)


def catalogue():
    """The action list as prompt text."""
    return "\n".join(f"- {name}: {purpose}" for name, (_, purpose) in ACTIONS.items())


def run(name, ctx):
    """Run the handler behind `name`. Returns its FACTS, or None if unknown.

    `ctx` carries the raw utterance, which is what the handlers parse — the
    model's job here is to say WHICH action, not to re-extract the details. That
    keeps one parser per feature instead of two that can disagree.
    """
    entry = ACTIONS.get(name)
    if entry is None:
        return None
    label, _purpose = entry

    # Imported here, not at module scope: dispatch imports responder, which
    # imports this module, so a top-level import would be a cycle.
    from intent.dispatch import dispatch
    from intent.intent import slots_for
    from intent.normalize import normalize

    slots = slots_for(label, normalize(ctx.query or ""))
    print(f"🔨 Action: {name} -> {label}")
    try:
        return dispatch(label, slots, ctx)
    except Exception as exc:               # noqa: BLE001
        # Same posture as tools.run: a broken handler must not take the whole
        # reply down, and the failure is reported as facts so the model says it
        # couldn't rather than inventing a success.
        print(f"⚠  action {name} failed: {type(exc).__name__}: {exc}")
        return f"{name}: failed"
