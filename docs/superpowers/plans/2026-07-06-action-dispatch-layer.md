# Action-Dispatch Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the if/elif action chain with an explicit, testable action-handler layer whose interface survives the later ML intent classifier and external integrations.

**Architecture:** Each action lives in its own file under `src/actions/` exporting a pure `handle(slots, ctx) -> str`. `intent/intent.py` returns a `(intent, slots)` tuple; `intent/dispatch.py` maps intent name → handler via an explicit dict and speaks nothing itself. Handlers return text; a single layer (`main`) speaks it; async actions use `ctx.speak(...)` for deferred output.

**Tech Stack:** Python 3.11 stdlib only (`dataclasses`, `datetime`, `threading`, `re`, `ast`, `operator`). No new dependencies.

## Global Constraints

- **Run from `src/`.** All imports use the `package.module` form (e.g. `from actions.time_action import handle`). Verification commands `cd src` first.
- **Local-first.** No network or cloud dependencies. Stdlib only for this feature.
- **Handlers must not import `audio.tts`.** Actions are decoupled from TTS — they return text and use `ctx.speak(...)` for deferred/async speech. Only `main.py` imports `say`.
- **Safe arithmetic only.** `calc_action` must never call bare `eval()`. Use an AST evaluator restricted to numeric arithmetic operators.
- **Exceptions surface.** `dispatch()` does NOT wrap handlers in try/except — handler bugs are visible during development.
- **Understandable over clever, one concern per file.** No decorators, no plugin autoloader, no class hierarchy — explicit registry dict only.
- **No committed test suite.** The repo has no test harness (`tests/` is gitignored scratch). Each task verifies with a runnable `python -c` check; nothing uncommitted is left behind.
- **Intent output shape is fixed:** `decide_action(query)` returns `(intent: str | None, slots: dict)`. This is what the Block 3 classifier will emit — do not change it.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/utils/context.py` | **Create.** `Context` dataclass carrying `speak` callback + raw `query`. |
| `src/actions/__init__.py` | **Create.** Empty package marker. |
| `src/actions/time_action.py` | **Create.** `handle` → current time string (computed at call time). |
| `src/actions/date_action.py` | **Create.** `handle` → current date string. |
| `src/actions/timer_action.py` | **Create.** `handle` → honors `duration` slot; deferred speech via `ctx.speak`. |
| `src/actions/calc_action.py` | **Create.** `handle` → safe arithmetic on `expression` slot. |
| `src/intent/intent.py` | **Rewrite.** Keyword matcher returning `(intent, slots)`; no TTS imports. |
| `src/intent/dispatch.py` | **Create.** `REGISTRY` dict + `dispatch()` + `respond()`. |
| `src/main.py` | **Rewrite.** Wire pipeline; build `Context`; guard loop with `__main__`. |
| `src/utils/utils.py` | **Delete.** Handlers move into `actions/`; bugs fixed there. |

---

### Task 1: Context dataclass

**Files:**
- Create: `src/utils/context.py`

**Interfaces:**
- Produces: `Context(speak: Callable[[str], None], query: str = "")` — dataclass with fields `speak` and `query`.

- [ ] **Step 1: Write the failing check**

Run: `cd src && python -c "from utils.context import Context; c = Context(speak=lambda s: None, query='hi'); print(c.query)"`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.context'`

- [ ] **Step 2: Create the file**

```python
# src/utils/context.py
from dataclasses import dataclass
from typing import Callable


@dataclass
class Context:
    """Passed to every action handler. Grows over time (e.g. conversation
    history in Block 4). `speak` lets async actions produce deferred speech."""
    speak: Callable[[str], None]
    query: str = ""
```

- [ ] **Step 3: Run the check to verify it passes**

Run: `cd src && python -c "from utils.context import Context; c = Context(speak=lambda s: None, query='hi'); print(c.query)"`
Expected: prints `hi`

- [ ] **Step 4: Commit**

```bash
git add src/utils/context.py
git commit -m "Add Context dataclass for action handlers"
```

---

### Task 2: Time and date actions

Two near-identical pure handlers (no slots, no ctx use). Grouped because a reviewer would accept/reject them together.

**Files:**
- Create: `src/actions/__init__.py`
- Create: `src/actions/time_action.py`
- Create: `src/actions/date_action.py`

**Interfaces:**
- Produces: `time_action.handle(slots: dict, ctx) -> str`, `date_action.handle(slots: dict, ctx) -> str`.

- [ ] **Step 1: Write the failing check**

Run: `cd src && python -c "from actions.time_action import handle; print(handle({}, None))"`
Expected: FAIL with `ModuleNotFoundError: No module named 'actions'`

- [ ] **Step 2: Create the package marker**

```python
# src/actions/__init__.py
```
(empty file)

- [ ] **Step 3: Create the time action**

```python
# src/actions/time_action.py
from datetime import datetime


def handle(slots, ctx):
    # Computed at call time — the old tell_time() froze this at import.
    now = datetime.now()
    return f"The time is {now:%I:%M %p}"
```

- [ ] **Step 4: Create the date action**

```python
# src/actions/date_action.py
from datetime import datetime


def handle(slots, ctx):
    today = datetime.now()
    return f"Today is {today:%A, %B %d, %Y}"
```

- [ ] **Step 5: Run the checks to verify they pass**

Run: `cd src && python -c "from actions.time_action import handle as t; from actions.date_action import handle as d; print(t({}, None)); print(d({}, None))"`
Expected: two lines, e.g.
```
The time is 03:47 PM
Today is Monday, July 06, 2026
```
(exact values reflect the current clock)

- [ ] **Step 6: Commit**

```bash
git add src/actions/__init__.py src/actions/time_action.py src/actions/date_action.py
git commit -m "Add time and date action handlers"
```

---

### Task 3: Timer action (slots + async)

Fixes the `set_timer()` bug that ignored the requested duration (hardcoded 10s). Uses `ctx.speak` for deferred output.

**Files:**
- Create: `src/actions/timer_action.py`

**Interfaces:**
- Consumes: `ctx.speak(text: str)` from `Context` (Task 1) — but tested with a stand-in object exposing `.speak`.
- Produces: `timer_action.handle(slots: dict, ctx) -> str`. Reads `slots["duration"]` (seconds, defaults to 300).

- [ ] **Step 1: Write the failing check**

Run: `cd src && python -c "from actions.timer_action import handle; print(handle({'duration': 1}, None))"`
Expected: FAIL with `ModuleNotFoundError: No module named 'actions.timer_action'`

- [ ] **Step 2: Create the timer action**

```python
# src/actions/timer_action.py
from threading import Timer


def handle(slots, ctx):
    seconds = slots.get("duration", 300)
    # Fire deferred speech when the timer elapses — action stays non-blocking.
    Timer(seconds, lambda: ctx.speak("Timer finished!")).start()
    return f"Timer set for {seconds} seconds."
```

- [ ] **Step 3: Run the check to verify immediate + deferred behavior**

Run:
```bash
cd src && python -c "
from actions.timer_action import handle
import types, time
spoken = []
ctx = types.SimpleNamespace(speak=lambda s: spoken.append(s))
print(handle({'duration': 1}, ctx))
time.sleep(1.2)
assert spoken == ['Timer finished!'], spoken
print('deferred ok')
"
```
Expected:
```
Timer set for 1 seconds.
deferred ok
```

- [ ] **Step 4: Commit**

```bash
git add src/actions/timer_action.py
git commit -m "Add timer action honoring requested duration"
```

---

### Task 4: Calc action (safe arithmetic)

Evaluates arithmetic from the `expression` slot without `eval()`. Extracts the math substring from spoken text, normalizes spoken operators, and evaluates a restricted AST.

**Files:**
- Create: `src/actions/calc_action.py`

**Interfaces:**
- Produces: `calc_action.handle(slots: dict, ctx) -> str`. Reads `slots["expression"]` (raw string). Returns `"That's <n>."` or `"I couldn't work that out."`.

- [ ] **Step 1: Write the failing check**

Run: `cd src && python -c "from actions.calc_action import handle; print(handle({'expression': '12 * 4'}, None))"`
Expected: FAIL with `ModuleNotFoundError: No module named 'actions.calc_action'`

- [ ] **Step 2: Create the calc action**

```python
# src/actions/calc_action.py
import ast
import operator
import re

# Only these node/operator types are allowed — anything else is rejected.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Spoken operators → symbols. Surrounding spaces avoid matching inside words.
_WORDS = {
    " plus ": " + ",
    " minus ": " - ",
    " times ": " * ",
    " multiplied by ": " * ",
    " divided by ": " / ",
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def handle(slots, ctx):
    text = slots.get("expression", "").lower()
    for word, symbol in _WORDS.items():
        text = text.replace(word, symbol)
    # Pull out the longest run of math characters, dropping words like "what is".
    candidates = re.findall(r"[-+*/().%\d\s]+", text)
    math = max(candidates, key=len).strip() if candidates else ""
    try:
        tree = ast.parse(math, mode="eval")
        result = _eval(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return "I couldn't work that out."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"That's {result}."
```

- [ ] **Step 3: Run the check to verify arithmetic + safety**

Run:
```bash
cd src && python -c "
from actions.calc_action import handle
print(handle({'expression': '12 times 4'}, None))
print(handle({'expression': 'what is 3 plus 4'}, None))
print(handle({'expression': '10 divided by 4'}, None))
print(handle({'expression': 'os.system(\"ls\")'}, None))
"
```
Expected:
```
That's 48.
That's 7.
That's 2.5.
I couldn't work that out.
```
(Safety note: the AST evaluator only accepts numeric `Constant`/`BinOp`/`UnaryOp` nodes, so any call/name — e.g. `os.system(...)` or `__import__("os").system(...)` — is rejected and nothing executes. An input like `__import__(1)` has its letters stripped to `(1)` and evaluates to `1` — harmless, since no import ever runs.)

- [ ] **Step 4: Commit**

```bash
git add src/actions/calc_action.py
git commit -m "Add calc action with safe AST arithmetic"
```

---

### Task 5: Rewrite intent as a (intent, slots) matcher

Replaces the if/elif-that-calls-handlers with a pure classifier returning `(intent, slots)`. No TTS imports. Parses timer duration and passes the raw query as the calc expression.

**Files:**
- Modify (full rewrite): `src/intent/intent.py`

**Interfaces:**
- Produces: `decide_action(query: str) -> tuple[str | None, dict]`. Intent names: `"TIME"`, `"DATE"`, `"TIMER"` (slot `duration` in seconds), `"CALC"` (slot `expression`), or `(None, {})`.

- [ ] **Step 1: Write the failing check (current output is a bare string)**

Run: `cd src && python -c "from intent.intent import decide_action; print(decide_action('what is the time'))"`
Expected: with the OLD code this speaks and prints `time` (a string, not a tuple). This confirms the pre-change behavior we are replacing.

- [ ] **Step 2: Rewrite the file**

```python
# src/intent/intent.py
import re

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}
_CALC_WORDS = (" plus ", " minus ", " times ", " divided by ", " multiplied by ")


def _parse_duration(query):
    """Pull '<n> second|minute|hour' out of the query; default 5 minutes."""
    match = re.search(r"(\d+)\s*(second|minute|hour)", query)
    if not match:
        return 300
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def decide_action(query):
    q = query.lower()
    if "time" in q and ("what" in q or "tell" in q):
        return ("TIME", {})
    if "date" in q or "what day" in q:
        return ("DATE", {})
    if "timer" in q:
        return ("TIMER", {"duration": _parse_duration(q)})
    if "calculate" in q or any(word in q for word in _CALC_WORDS):
        return ("CALC", {"expression": q})
    return (None, {})
```

- [ ] **Step 3: Run the check to verify the new tuple output**

Run:
```bash
cd src && python -c "
from intent.intent import decide_action
print(decide_action('what is the time'))
print(decide_action('set a timer for 5 minutes'))
print(decide_action('what is 3 plus 4'))
print(decide_action('I like pizza'))
"
```
Expected:
```
('TIME', {})
('TIMER', {'duration': 300})
('CALC', {'expression': 'what is 3 plus 4'})
(None, {})
```

- [ ] **Step 4: Commit**

```bash
git add src/intent/intent.py
git commit -m "Rewrite intent to return (intent, slots); no TTS coupling"
```

---

### Task 6: Dispatch + respond

The explicit registry and the query→reply glue. Imports actions and intent — no audio imports, so it stays testable without TTS.

**Files:**
- Create: `src/intent/dispatch.py`

**Interfaces:**
- Consumes: `decide_action` (Task 5); `handle` from each action module (Tasks 2–4).
- Produces:
  - `dispatch(intent, slots, ctx) -> str | None` — returns `None` when no handler is registered.
  - `respond(query: str, ctx) -> str` — full pipeline; returns handler text or the fallback line.

- [ ] **Step 1: Write the failing check**

Run: `cd src && python -c "from intent.dispatch import respond; print('ok')"`
Expected: FAIL with `ModuleNotFoundError: No module named 'intent.dispatch'`

- [ ] **Step 2: Create the dispatch module**

```python
# src/intent/dispatch.py
from actions import time_action, date_action, timer_action, calc_action
from intent.intent import decide_action

REGISTRY = {
    "TIME": time_action.handle,
    "DATE": date_action.handle,
    "TIMER": timer_action.handle,
    "CALC": calc_action.handle,
}


def dispatch(intent, slots, ctx):
    """Look up and run the handler. Returns None when nothing is registered.
    Handler exceptions are intentionally NOT caught — bugs stay visible."""
    handler = REGISTRY.get(intent)
    if handler is None:
        return None
    return handler(slots, ctx)


def respond(query, ctx):
    """Full pipeline: query -> intent -> handler text (or fallback line)."""
    intent, slots = decide_action(query)
    reply = dispatch(intent, slots, ctx)
    if reply is None:
        return "Sorry, I didn't catch that."
    return reply
```

- [ ] **Step 3: Run the check to verify dispatch + fallback**

Run:
```bash
cd src && python -c "
from intent.dispatch import respond
import types
ctx = types.SimpleNamespace(speak=lambda s: None, query='')
print(respond('what is the time', ctx))
print(respond('blah blah blah', ctx))
"
```
Expected:
```
The time is 03:47 PM
Sorry, I didn't catch that.
```
(first line reflects the current clock)

- [ ] **Step 4: Commit**

```bash
git add src/intent/dispatch.py
git commit -m "Add explicit action registry and dispatch/respond"
```

---

### Task 7: Wire main.py and remove utils.py

Rewrites the loop to use `respond` + `Context`, guards the loop with `__main__` so the module is importable, and deletes the now-dead `utils/utils.py`.

**Files:**
- Modify (full rewrite): `src/main.py`
- Delete: `src/utils/utils.py`

**Interfaces:**
- Consumes: `record` (audio.audio), `transcribe` (audio.stt), `say` (audio.tts), `respond` (Task 6), `Context` (Task 1).

- [ ] **Step 1: Confirm nothing else imports utils.utils**

Run: `cd src && grep -rn "utils.utils\|from utils import utils" . || echo "no references"`
Expected: only `main.py` (which we are rewriting) appears, or `no references`. If any OTHER module references it, stop and update that module first.

- [ ] **Step 2: Rewrite main.py**

```python
# src/main.py
from audio.audio import record
from audio.stt import transcribe
from audio.tts import say
from intent.dispatch import respond
from utils.context import Context


def main():
    print("JANET is running. Press Ctrl-C to quit.")
    while True:
        audio = record()
        query = transcribe(audio)
        print(f"🗣  You said: {query.strip()}")
        ctx = Context(speak=say, query=query)
        reply = respond(query, ctx)
        print(f"⚙️  Reply: {reply}")
        say(reply)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Delete the dead handlers file**

Run: `git rm src/utils/utils.py`
Expected: `rm 'src/utils/utils.py'`

- [ ] **Step 4: Verify main.py compiles without importing the audio stack**

Run: `cd src && python -m py_compile main.py && echo "compiles ok"`
Expected: prints `compiles ok` (a compile check avoids initializing pyttsx3/pyaudio; the runtime pipeline was already verified via `respond` in Task 6).

- [ ] **Step 5: Commit**

```bash
git add src/main.py
git commit -m "Wire main to dispatch pipeline; remove dead utils.py"
```

---

## Manual end-to-end check (after all tasks)

With a mic/speaker available, from `src/` run `python main.py`, then hold SPACE and speak:

- "What is the time" → speaks the current time.
- "What is the date" → speaks today's date.
- "Set a timer for 5 seconds" → confirms, then says "Timer finished!" ~5s later.
- "What is twelve times four" → speaks "That's 48." (Whisper must transcribe digits; if it writes words, calc handles "twelve" poorly — expected limitation, noted for a later pass.)
- "I like pizza" → "Sorry, I didn't catch that."

This is a smoke test, not a gate — the per-task checks already validated each unit.

## Notes for the executor

- **Spoken-number limitation:** `calc_action` normalizes spoken *operators* ("plus", "times") but not spoken *numbers* ("twelve"). Digit input works; word-number input is out of scope for this pass. Do not expand it here.
- **Timer message grammar:** `"Timer set for 1 seconds."` is intentionally left un-pluralized — keep it simple; pluralization is not worth the code here.
- **Uncommitted baseline note:** this worktree branched from clean `origin/master`. If the owner had in-progress edits to `intent.py`/`utils.py` on the main checkout, this plan's rewrites supersede them — reconcile before merging.
