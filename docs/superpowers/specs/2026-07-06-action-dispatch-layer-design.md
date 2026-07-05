# Action-Dispatch Layer — Design

**Date:** 2026-07-06
**Status:** Approved (design), pending implementation plan
**Scope:** Internal action dispatch only. External integrations and the ML intent classifier are explicitly out of scope and come later.

## Context

`src/` today is a thin push-to-talk slice: `record → transcribe → decide_action → say`. Action selection is a hardcoded if/elif chain in `intent/intent.py` that directly imports and calls two functions (`tell_time`, `set_timer`). There is no dispatch layer, handler contract, or registration mechanism — "improve the integration system" is really "build one for the first time."

Agreed sequencing with the owner:

1. **Dispatch architecture (this spec)** — build the internal action layer.
2. **ML intent** (Block 3) — DistilBERT classifier replaces the keyword matcher.
3. **External integrations** — weather, calendar, smart home, built on top of this layer.

This spec is deliberately shaped so steps 2 and 3 slot in without reworking the dispatch layer.

## Goals

- Replace the if/elif chain with an explicit, testable action-handler layer.
- One concern per file: each action in its own module (per CLAUDE.md working style).
- Decouple actions from TTS so they are testable without audio and survive the Block 5 threading model.
- Define an intent-output shape that matches what the Block 3 classifier will emit, so `dispatch.py` never has to change when the classifier lands.
- Fix two latent bugs in the current handlers along the way.

## Non-goals (YAGNI)

- No decorator auto-registration, no plugin auto-loader, no class hierarchy — explicit dict only.
- No external/network integrations (weather, calendar) in this pass. Local-first only.
- No formal test suite (the repo has none; `tests/` is gitignored scratch). The return-text contract makes actions unit-testable, which is enough.
- No LLM fallback wiring yet — but the dispatch design leaves a clean seam for it.

## Design decisions

Two decisions were settled during brainstorming:

1. **Registration = explicit registry dict.** Each action lives in its own file exporting a `handle` function. One dispatch module imports them and maps intent name → handler in a plain dict. Chosen over decorator auto-registration (import-time-side-effect magic) and class-based handlers (OOP ceremony) because it is the most explainable — every line is visible, nothing happens by magic. Cost: adding an action touches two places (new file + one registry line), which is acceptable and explicit.

2. **Handlers return text; `ctx` carries a `speak` callback.** Handlers return a string that a single layer (`main`) speaks. Async actions (timer) get `ctx.speak(...)` for deferred output when they fire later. Chosen over handlers calling `say()` directly because it keeps actions pure and testable and won't fight threading in Block 5.

Two smaller calls: handler exceptions **surface** (not swallowed) so bugs are visible during development; and **CALC is included** in the starter set to prove a non-trivial handler drops in cleanly.

## File layout

```
src/actions/                 ← new package, one file per action
   __init__.py
   time_action.py            handle(slots, ctx) -> str
   date_action.py            handle(slots, ctx) -> str
   timer_action.py           handle(slots, ctx) -> str   (slots + async)
   calc_action.py            handle(slots, ctx) -> str   (safe arithmetic)
src/intent/
   intent.py                 keyword matcher → returns (intent, slots)
   dispatch.py               REGISTRY dict + dispatch(intent, slots, ctx)
src/utils/
   context.py                the Context dataclass
```

`src/utils/utils.py` is emptied — its handlers move into `src/actions/` and get their bugs fixed on the way.

Imports stay in the existing `package.module` form (run from inside `src/`), e.g. `from actions.time_action import handle`.

## Data shapes

### Intent output

`intent.py`'s `decide_action(query)` stops returning a bare string and returns `(intent, slots)`:

```python
("TIME",  {})
("DATE",  {})
("TIMER", {"duration": 300})   # parsed "5 minutes" → seconds
("CALC",  {"expression": "12 * 4"})
(None,    {})                  # no match → LLM fallback seam
```

Intent names are uppercase string constants. This `(label, slots)` shape is exactly what the Block 3 DistilBERT classifier will emit, so only `intent.py`'s internals change in Block 3 — `dispatch.py` and every handler stay put.

### Context

A tiny dataclass passed to every handler, with room to grow (Block 4 context window / conversation history land here later):

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Context:
    speak: Callable[[str], None]   # = say; for deferred/async speech
    query: str                     # raw transcript (for LLM fallback later)
```

## Handler contract

```python
def handle(slots: dict, ctx: Context) -> str:
    """Return text to speak. Use ctx.speak(...) for deferred/async output."""
```

- **Sync actions** (time, date, calc) return a string; `main` speaks it.
- **Async actions** (timer) return an immediate confirmation string *and* schedule `ctx.speak(...)` for when the timer fires.

Example async handler:

```python
# actions/timer_action.py
from threading import Timer

def handle(slots, ctx):
    secs = slots.get("duration", 300)
    Timer(secs, lambda: ctx.speak("Timer finished!")).start()
    return f"Timer set for {secs} seconds."
```

## Dispatch + main loop

```python
# intent/dispatch.py
from actions import time_action, date_action, timer_action, calc_action

REGISTRY = {
    "TIME":  time_action.handle,
    "DATE":  date_action.handle,
    "TIMER": timer_action.handle,
    "CALC":  calc_action.handle,
}

def dispatch(intent, slots, ctx):
    handler = REGISTRY.get(intent)
    return handler(slots, ctx) if handler else None   # None → fallback seam
```

```python
# main.py loop
from utils.context import Context

intent, slots = decide_action(query)
reply = dispatch(intent, slots, Context(speak=say, query=query))
if reply:
    say(reply)
else:
    say("Sorry, I didn't catch that.")   # ← Ollama LLM fallback plugs in here
```

Handler exceptions are **not** caught by dispatch — they surface and are visible during development. (A friendly catch-all can be added later once the action set stabilizes.)

## Bugs fixed as part of this work

- **`tell_time`** currently computes `now = datetime.now()` once at module import, so it always reports the time the program *started*. The new `time_action.handle` computes `datetime.now()` at call time.
- **`set_timer`** currently ignores the requested duration and hardcodes `Timer(10.0, ...)`. The new `timer_action.handle` honors the parsed `duration` slot (defaulting to a sensible value if absent).

## Starter action set

| Intent | Slots | Notes |
|--------|-------|-------|
| TIME   | —     | Current time, computed at call time. |
| DATE   | —     | Current date. |
| TIMER  | `duration` (seconds) | Parses "5 minutes" etc.; async via `ctx.speak`. |
| CALC   | `expression` | Safe arithmetic evaluation (no arbitrary `eval`). |

CALC must evaluate arithmetic **safely** — not via bare `eval()`. Implementation plan will pick a safe approach (e.g. an AST-based numeric evaluator restricted to arithmetic operators).

## How this survives later steps

- **Block 3 (ML intent):** only `intent.py` changes; it already returns `(label, slots)`. Dispatch and handlers untouched.
- **LLM fallback:** the `None` branch in the main loop is the seam — route unmatched queries to Ollama there.
- **External integrations:** new connectors are just new files in `actions/` plus one registry line each — the same pattern, no architectural change.
- **Block 5 (threading):** actions are pure functions returning text; the speak side-effect is already isolated behind `ctx.speak`, so moving TTS to its own thread is a localized change.
