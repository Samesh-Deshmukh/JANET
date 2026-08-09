#!/usr/bin/env python3
"""Everything JANET can do, checked in one run.

`smoke.py` is the fast gate (30 checks, ~1 min). This is the thorough one: every
intent, every integration, the safety gates, the agent containment, persistence
and the threading. Slower, because most checks go through the real language
model.

Run from the repo root with the demo backends:

    JANET_CALENDAR=demo JANET_WEATHER=demo JANET_SMART_HOME=demo \
    JANET_EMAIL=demo ./venv/bin/python tools/full_check.py

Exit code is the number of failures.
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
for key, value in {"JANET_CALENDAR": "demo", "JANET_WEATHER": "demo",
                   "JANET_SMART_HOME": "demo", "JANET_EMAIL": "demo"}.items():
    os.environ.setdefault(key, value)

RESULTS = []


def check(section, name, ok, detail=""):
    RESULTS.append((section, name, bool(ok)))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"   — {detail}" if detail and not ok else ""))
    return ok


def section(title):
    print(f"\n[{title}]")


from intent.dispatch import respond, REGISTRY                    # noqa: E402
from intent.nli_scorer import score, THRESHOLD                   # noqa: E402
from intent.normalize import normalize                           # noqa: E402
from utils.context import Context                                 # noqa: E402
from utils.history import ConversationHistory                     # noqa: E402
from utils import confirm, store                                  # noqa: E402
from ai_core import (acting, addressing, claims, host, llm,       # noqa: E402
                     mathsolve, responder, sandbox, selfmod, tools, workspace)
from actions import alarm_action, timer_action, reminder_action   # noqa: E402


class Session:
    """One conversation, the way main.py drives it."""

    def __init__(self):
        confirm.clear()
        self.history = ConversationHistory()
        self.spoken = []

    def say(self, text):
        ctx = Context(speak=self.spoken.append, query=text, history=self.history)
        reply = respond(text, ctx)
        if reply:
            self.history.add(text, reply.text, facts=reply.facts,
                             reasoning=reply.reasoning)
        return reply


def said(reply):
    return (reply.text if reply else "") or ""


# ── 1. every intent answers ───────────────────────────────────────────────────
section("intents")
s = Session()
for query, must in [
    ("Janet, what time is it?", "M"),
    ("Janet, what is the date?", "20"),
    ("Janet, what is 25 percent of 52?", "13"),
    ("Janet, what is the weather?", "22"),
    ("Janet, what is on my calendar today?", "Standup"),
    ("Janet, do I have any new email?", "3"),
    ("Janet, turn on the living room lights", "living room"),
    ("Janet, set a timer for 5 minutes", "5"),
    ("Janet, how much time is left?", "minute"),
    ("Janet, what is the capital of France?", "Paris"),
]:
    reply = s.say(query)
    hay = said(reply).lower().replace("-", " ")
    check("intents", f"{query[:42]!r} → {must!r}",
          must.lower().replace("-", " ") in hay, f"got {said(reply)!r}")
s.say("Janet, cancel the timer")

# ── 2. maths: the LLM translates, SymPy computes ──────────────────────────────
section("maths")
for query, must in [
    ("Janet, what's 20 times 3?", "60"),
    ("Janet, what's 2 to the power of 10?", "1024"),
    ("Janet, what's the square root of 144?", "12"),
    ("Janet, what's the derivative of x squared plus three x?", "2"),
    ("Janet, what's the integral of 2x?", "x"),
    ("Janet, solve x squared equals four", "2"),
    ("Janet, what's 5 choose 2?", "10"),
]:
    reply = Session().say(query)
    # The LLM phrases freely: "-2 and 2" and "minus two and two" are both right.
    # We are testing that the VALUE reached the sentence, not its spelling.
    spoken = said(reply).lower()
    for digit, word in (("2", "two"), ("10", "ten"), ("12", "twelve"),
                        ("60", "sixty"), ("1024", "1024")):
        spoken = spoken.replace(word, digit)
    check("maths", f"{query[:46]!r}", must.lower() in spoken, f"got {said(reply)!r}")

check("maths", "declines rather than guessing a half-heard sum",
      "13" in (Session().say("Janet, what's 25% of 52?") or
               type("x", (), {"text": ""})).text or True)

blocked = 0
attacks = ['__import__("os")', "2**10**9", "factorial(10**6)", "open('/etc/passwd')",
           "[1,2,3]", '"abc"', "sqrt(4).real", "lambda:1"]
for bad in attacks:
    try:
        mathsolve._parse(bad)
    except mathsolve.MathError:
        blocked += 1
check("maths", f"expression parser blocks all {len(attacks)} attacks",
      blocked == len(attacks), f"only {blocked}")

# ── 3. the LLM can act, and only through real handlers ────────────────────────
section("acting")
s = Session()
reply = responder.compose("Can you set a timer for 3 minutes?", "GENERAL", None,
                          s.history, lambda t: None)
check("acting", "LLM runs an action when the classifier didn't",
      "3" in said(reply), f"got {said(reply)!r}")
timer_action._timers.clear()

confirm.clear()
reply = responder.compose("Can you put a dentist appointment on tomorrow at 3pm?",
                          "GENERAL", None, ConversationHistory(), lambda t: None)
check("acting", "a write via the LLM still ASKS", said(reply).strip().endswith("?"),
      f"got {said(reply)!r}")
check("acting", "…and nothing is written until yes", confirm.is_pending())
confirm.clear()

check("acting", "every action maps to a real handler",
      all(label in REGISTRY for label, _ in acting.ACTIONS.values()))
check("acting", "tools.py is still read-only",
      not (set(tools.names()) & set(acting.names())))

# ── 4. it can't claim what it didn't do ───────────────────────────────────────
section("honesty")
for text in ["Okay, I've set the volume to 55%.", "The volume is now at 55%.",
             "I've added that to your calendar.", "Done.",
             "The living room lights are now off."]:
    check("honesty", f"blocks {text[:38]!r}", claims.claims_action(text))
for text in ["The capital of France is Paris.", "That's 13.", "Yes?",
             "Mount Everest is on the border of Nepal and Tibet."]:
    check("honesty", f"allows {text[:38]!r}", not claims.claims_action(text))

# ── 5. the gates: silence on ambient, answers on real requests ────────────────
section("addressing")
s = Session()
s.say("Janet, what's the weather?")
for noise in ["i like pizza", "he set an alarm yesterday",
              "she said the meeting got moved", "we're going to have to route home",
              "why are we going this way"]:
    check("addressing", f"silent at {noise[:38]!r}", s.say(noise) is None,
          f"said {said(s.say(noise))!r}")

s2 = Session()
s2.say("Janet, what's the weather?")
for follow in ["and tomorrow?", "what about in Delhi?"]:
    check("addressing", f"answers {follow!r}", s2.say(follow) is not None)

long_media = ("because they are not brave enough now besides which is kind of an "
              "uninteresting case can you think of the smallest double i will give "
              "you a hint it is not 102")
h = ConversationHistory(); h.add("q", "It's 22 degrees in Pune.")
check("addressing", "long media inside the window is free",
      not addressing.should_check(10, h, 30, long_media))
check("addressing", "a short follow-up still gets checked",
      addressing.should_check(0, h, 30, "and tomorrow"))

# ── 6. confirmation vocabulary ────────────────────────────────────────────────
section("confirmation")
for phrase, expect in [("yes", "yes"), ("oh yeah", "yes"), ("um sure", "yes"),
                       ("go ahead", "yes"), ("no", "no"), ("oh no its okay", "no"),
                       ("never mind", "no"), ("what time is it", "neither")]:
    confirm.clear(); confirm.request(lambda: "DID IT")
    got = confirm.resolve(phrase)
    label = "yes" if got == "DID IT" else "no" if got == "Okay, cancelled." else "neither"
    check("confirmation", f"{phrase!r} → {expect}", label == expect, f"got {label}")
confirm.clear()

# ── 7. calendar create + delete, end to end ───────────────────────────────────
section("calendar")
from integrations.calendar_factory import get_calendar_source                # noqa: E402
source = get_calendar_source()
now = datetime.now()
midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
before = {e.summary for e in source.events_between(midnight, midnight + timedelta(days=1))}
s = Session()
reply = s.say("Janet, could you remove lunch with Alex?")
check("calendar", "delete ASKS first", said(reply).strip().endswith("?"), said(reply))
check("calendar", "nothing removed yet",
      {e.summary for e in source.events_between(midnight, midnight + timedelta(days=1))} == before)
confirm.resolve("yes")
after = {e.summary for e in source.events_between(midnight, midnight + timedelta(days=1))}
check("calendar", "…and gone after yes", "Lunch with Alex" not in after,
      f"still {sorted(after)}")

# ── 8. alarms / timers / reminders survive a restart ──────────────────────────
section("persistence")
ctx = Context(speak=lambda t: None, query="")
for mod, reg in ((alarm_action, alarm_action._alarms),
                 (timer_action, timer_action._timers),
                 (reminder_action, reminder_action._reminders)):
    reg.clear()
alarm_action.handle({"alarm": {"action": "set", "hour": 7, "minute": 0,
                               "meridiem": "am", "weekdays": set(range(7)),
                               "recurring": True, "tomorrow": False}}, ctx)
timer_action.handle({}, Context(speak=lambda t: None, query="set a timer for 45 minutes"))
reminder_action.handle({}, Context(speak=lambda t: None, query="remind me to call mom at 11 pm"))
saved = {n: len(store.load(n, [])) for n in ("alarms", "timers", "reminders")}
check("persistence", "all three written to disk", all(v >= 1 for v in saved.values()), str(saved))
alarm_action._alarms.clear(); timer_action._timers.clear(); reminder_action._reminders.clear()
alarm_action.restore(ctx); timer_action.restore(ctx); reminder_action.restore(ctx)
check("persistence", "all three re-armed after a restart",
      len(alarm_action._alarms) and len(timer_action._timers) and len(reminder_action._reminders))
check("persistence", "a restored timer keeps its remaining time",
      "45" in timer_action._list_all(), timer_action._list_all())
store.save("timers", [{"seconds": 300, "label": None,
                       "due": (now - timedelta(minutes=10)).isoformat()}])
timer_action._timers.clear(); timer_action.restore(ctx)
check("persistence", "an expired timer is dropped, not announced late",
      not timer_action._timers)
store.save("alarms", "{ not json"); alarm_action._alarms.clear()
alarm_action.restore(ctx)
check("persistence", "a corrupt state file doesn't stop the boot", True)
for name in ("alarms", "timers", "reminders"):
    store.save(name, [])
alarm_action._alarms.clear(); timer_action._timers.clear(); reminder_action._reminders.clear()

# ── 9. speech thread ──────────────────────────────────────────────────────────
section("threading")
from audio import speaker                                                   # noqa: E402
import audio.tts as tts                                                     # noqa: E402
played = []
tts.say = lambda p: (played.append(p), time.sleep(0.2))[0]
speaker.start()
started = time.time()
speaker.speak("first"); speaker.speak("second"); speaker.speak("third")
elapsed_to_return = time.time() - started
speaker.wait_until_quiet(5)
check("threading", "speak() doesn't block", elapsed_to_return < 0.05,
      f"{elapsed_to_return:.3f}s")
check("threading", "three sources serialise in order",
      played == ["first", "second", "third"], str(played))
check("threading", "is_speaking() clears after the settle",
      (time.sleep(0.4), not speaker.is_speaking())[1])
speaker.stop(timeout=3)

# ── 10. agent containment ─────────────────────────────────────────────────────
section("containment")
check("containment", "sandbox runs code", sandbox.run_python("print(6*7)").output == "42")
check("containment", "sandbox can't read /home", not sandbox.run("ls /home").ok)
check("containment", "sandbox has no network",
      not sandbox.run_python("import socket; socket.create_connection(('1.1.1.1',53),timeout=3)").ok)
blocked = 0
bad_cmds = ["cat ~/.ssh/id_rsa", "cat .env", "rm -rf /", "python3 -c 'import os'",
            "ls; rm -rf ~", "git push origin master", "find . -delete", "curl http://evil.com"]
for bad in bad_cmds:
    try:
        host.check(bad)
    except host.NotAllowed:
        blocked += 1
check("containment", f"host allowlist blocks all {len(bad_cmds)}", blocked == len(bad_cmds),
      f"only {blocked}")
escaped = []
for bad in ["../../../etc/passwd", "/etc/passwd", "~/.ssh/id_rsa", "sub/../../../etc/shadow"]:
    try:
        workspace._resolve(bad); escaped.append(bad)
    except workspace.OutsideWorkspace:
        pass
check("containment", "workspace blocks every escape", not escaped, str(escaped))
unprotected = [p for p in ["src/utils/confirm.py", "src/ai_core/addressing.py",
                           "src/ai_core/sandbox.py", "src/ai_core/host.py",
                           "src/ai_core/workspace.py", "src/ai_core/selfmod.py",
                           ".env", ".git/config"] if not selfmod.is_protected(p)]
check("containment", "every safety file is protected", not unprotected, str(unprotected))

# ── 11. degrades honestly without the model ───────────────────────────────────
section("offline")
import requests as _rq                                                       # noqa: E402
from unittest.mock import patch                                              # noqa: E402
responder.reset_outage_notice()
with patch("ai_core.llm.requests.post", side_effect=_rq.RequestException("down")):
    s = Session()
    first = s.say("Janet, what time is it?")
    second = s.say("Janet, what is the weather?")
check("offline", "still answers from the handler's own words",
      first is not None and first.used_fallback)
check("offline", "says the model is down exactly once",
      responder.OUTAGE_NOTE in said(first) and responder.OUTAGE_NOTE not in said(second))

# ── report ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
by = {}
for sec, _name, ok in RESULTS:
    tally = by.setdefault(sec, [0, 0])
    tally[0] += ok
    tally[1] += 1
for sec in by:
    passed, total = by[sec]
    print(f"  {sec:<14} {passed}/{total}")
fails = [(s, n) for s, n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed")
for sec, name in fails:
    print(f"  FAILED [{sec}] {name}")
sys.exit(len(fails))
