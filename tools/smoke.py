#!/usr/bin/env python3
"""Smoke test for JANET's pipeline — the checks that were being done by hand.

Run it from anywhere:

    venv/bin/python tools/smoke.py

Every integration is switched to its demo backend, so this needs no accounts, no
credentials and no network. It exercises the REAL pipeline (`dispatch.respond`),
not mocks, which is the point: it catches "JANET no longer answers" rather than
"this function returns what it returned yesterday".

Why it exists: JANET can now modify its own source (see `ai_core/selfmod.py`),
and "extensive testing before merge" needs something to run. This is that thing.
It is also just useful — it is the check to run before pushing anything.

Exit code 0 = everything passed. Non-zero = the number of failures.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Demo backends: believable data, no credentials, no network.
os.environ.setdefault("JANET_CALENDAR", "demo")
os.environ.setdefault("JANET_WEATHER", "demo")
os.environ.setdefault("JANET_SMART_HOME", "demo")
os.environ.setdefault("JANET_EMAIL", "demo")

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not condition else ""))
    return bool(condition)


def main():
    from intent.dispatch import respond, REGISTRY
    from utils.context import Context
    from utils.history import ConversationHistory
    from utils import confirm

    print("JANET smoke test\n" + "=" * 60)

    # --- 1. the app is wired together at all ------------------------------
    print("\n[wiring]")
    import main as janet_main                                    # noqa: F401
    check("main.py imports", True)
    check("every intent has a handler", len(REGISTRY) == 12,
          f"registry has {len(REGISTRY)}")

    # --- 2. the two gates still silence ambient speech --------------------
    # The single most important property: an always-on mic that answers the
    # television is worse than one that misses a question.
    print("\n[silence]")
    confirm.clear()
    history = ConversationHistory()
    for phrase in ("i like pizza", "the weather is nice today",
                   "he set an alarm yesterday"):
        reply = respond(phrase, Context(speak=lambda s: None, query=phrase,
                                        history=history))
        check(f"silent on {phrase!r}", reply is None,
              f"answered {reply.text!r}" if reply else "")

    # --- 3. each capability answers, and the value survives ---------------
    print("\n[capabilities]")
    history = ConversationHistory()

    def ask(query, must_contain):
        ctx = Context(speak=lambda s: None, query=query, history=history)
        reply = respond(query, ctx)
        text = reply.text if reply else None
        # Compare with hyphens as spaces: the LLM phrases freely, so "a 5-minute
        # timer" and "5 minutes" are both correct answers and neither should be
        # a failure. We're testing that the right VALUE reached the sentence.
        haystack = (text or "").lower().replace("-", " ")
        ok = text is not None and must_contain.lower().replace("-", " ") in haystack
        check(f"{query[:44]!r} mentions {must_contain!r}", ok, f"got {text!r}")
        if reply:
            history.add(query, reply.text, facts=reply.facts,
                        reasoning=reply.reasoning)
        return reply

    ask("Janet, what time is it?", "M")               # 4:49 PM / AM
    ask("Janet, what is the date?", "20")             # a year
    ask("Janet, what is 25 percent of 52?", "13")
    ask("Janet, what is the weather?", "22")
    ask("Janet, what is on my calendar today?", "Standup")
    ask("Janet, do I have any new email?", "3")
    ask("Janet, turn on the living room lights", "living room")
    ask("Janet, set a timer for 5 minutes", "5 minute")
    ask("Janet, how much time is left?", "minute")
    ask("Janet, cancel the timer", "cancel")
    ask("Janet, what is the capital of France?", "Paris")

    # --- 4. writes still ask first ---------------------------------------
    # If this regresses, JANET silently writes to a real calendar.
    print("\n[confirmation]")
    confirm.clear()
    reply = ask("Janet, schedule a review tomorrow at 2 pm", "?")
    check("a write action ASKS rather than acting",
          reply is not None and reply.text.strip().endswith("?"),
          f"got {reply.text!r}" if reply else "no reply")
    check("something is pending confirmation", confirm.is_pending())
    confirm.clear()

    # --- 5. the containment layers still contain -------------------------
    print("\n[containment]")
    from ai_core import sandbox, host, workspace
    if sandbox.available():
        check("sandbox runs code", sandbox.run_python("print(6*7)").output == "42")
        check("sandbox cannot read /home",
              not sandbox.run("ls /home").ok)
        check("sandbox has no network",
              not sandbox.run_python(
                  "import socket; socket.create_connection(('1.1.1.1',53),timeout=3)").ok)
    else:
        check("bwrap installed", False, "sandbox unavailable")

    for bad in ("cat ~/.ssh/id_rsa", "cat .env", "rm -rf /",
                "python3 -c 'import os'", "ls; rm -rf ~"):
        try:
            host.check(bad)
            check(f"host refuses {bad!r}", False, "it was ALLOWED")
        except host.NotAllowed:
            check(f"host refuses {bad!r}", True)

    for bad in ("../../../etc/passwd", "/etc/passwd", "~/.ssh/id_rsa"):
        try:
            workspace._resolve(bad)
            check(f"workspace refuses {bad!r}", False, "it ESCAPED")
        except workspace.OutsideWorkspace:
            check(f"workspace refuses {bad!r}", True)

    # --- report ----------------------------------------------------------
    failures = [(n, d) for n, ok, d in _results if not ok]
    print("\n" + "=" * 60)
    print(f"{len(_results) - len(failures)}/{len(_results)} passed")
    for name, detail in failures:
        print(f"  FAILED: {name}" + (f" — {detail}" if detail else ""))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
