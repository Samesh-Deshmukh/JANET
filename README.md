# JANET

**Just Another Neural Execution Tool** — a fully-local, privacy-first, wake-word-free voice assistant.

## What is JANET?

JANET is a voice assistant that runs **entirely on your own computer**. You talk
to it; it listens, works out whether you were talking to *it*, does the thing,
and answers out loud. No account, no subscription, no audio leaving the machine.

The difference from Alexa or Siri isn't just privacy — it's that **there's no
wake word.** You don't prefix every sentence with "Hey Janet." It listens all the
time and decides for itself whether a given sentence was meant for it. That's a
harder problem than it sounds, and most of this repo is the machinery for getting
it right: two cheap gates that stay silent by default, and a local language model
that gets the final say when they're unsure.

### What it's actually like

Every line below is real output, not a mock-up — captured by running the
conversation through the pipeline:

```
you   : Janet, set a timer for 10 minutes
JANET : I've set a timer for 10 minutes.

you   : how much time is left?                    ← no name needed
JANET : There's 9 minutes and 59 seconds left.

you   : I should probably set an alarm for tomorrow morning
JANET : (silence)                                 ← you weren't talking to it

you   : Janet, what's the weather?
JANET : It's 22 degrees and partly cloudy in Pune.

you   : and tomorrow?                             ← scores ZERO on both gates
JANET : Tomorrow, it will be rainy in Pune with a high of 27.6C and a low of 21.2C.
```

Three things are happening there that are worth naming:

- **"how much time is left?"** works without saying "Janet" — the sentence is
  shaped like a request, and that's enough.
- **The alarm sentence is ignored.** It contains the word "alarm" and it's about
  setting one, but you were thinking out loud. JANET checked and stayed quiet.
  For an always-on mic, *not answering* is the feature.
- **"and tomorrow?"** scores **0** — no question word, no command, no keyword,
  no name. Both gates would drop it. The language model is the only part of JANET
  holding the conversation, so it gets asked, recognises a follow-up, and calls
  the weather tool itself.

### What makes it different

| | |
|---|---|
| 🎙 **No wake word** | It works out when it's being addressed. Saying "Janet" helps, but plenty of sentences don't need it. |
| 🔒 **Nothing leaves your machine** | Speech recognition, intent, and the language model are all local. Only the optional calendar/weather/smart-home/email integrations touch the network, and each has a self-hostable or fake option. |
| 🤫 **Silence is the default** | Two gates must *both* agree before it says anything. An always-listening assistant that answers the television is worse than one that misses a question. |
| ✋ **It asks before anything irreversible** | Creating a calendar event or sending an email gets stated back and waits for a spoken yes. Some things — shutdown, door locks — it simply refuses to do at all. |
| 🚫 **It can't claim it did something it didn't** | If no action ran, no reply is allowed to say one did. This is enforced mechanically, not just asked for politely. |

### What it isn't

Being straight about scope: this is a **working prototype and a learning
project**, not a product. It has no installer, no GUI, and no packaging. It's
developed on Linux (WSL2 works; macOS and Windows need [two short files
ported](instructions.md#13-macos-windows-and-everything-else)). It wants a GPU
and about 10 GB of disk. Text-to-speech is still espeak-ng, which sounds like
1995 — Piper is the next job.

What *is* done: the full audio pipeline, both intent gates, all twelve intent
handlers, four integrations with fake backends, persistence, the confirmation
gate, and the agent layer.

## 📦 Install

**[instructions.md](instructions.md)** is the full setup guide — system packages
for every major distro plus macOS/Windows, the CUDA-vs-CPU split in
`requirements.txt`, training the intent classifier (**required** — the trained
model is gitignored, so it isn't in your clone), picking an LLM for your
hardware, and a troubleshooting section.

The 60-second version, for the impatient:

```bash
git clone https://github.com/Samesh-Deshmukh/JANET.git && cd JANET
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src && python -m intent.train    # ~16s on a GPU — you MUST do this
python main.py                      # then just talk
```

A microphone, a speaker, and ~10 GB of disk.
[Details, and what to do when it doesn't work →](instructions.md)

## Contents

**Start here**
| | |
|---|---|
| [How it works](#how-it-works) | How an utterance becomes a reply, and the two gates that can stop it |
| [Current capabilities](#current-capabilities) | Everything you can say to it |
| [Running it](#running-it) | Starting it, the LLM server, and reading its console output |

**How it's built**
| | |
|---|---|
| [The LLM does the talking](#the-llm-does-the-talking) | Why handlers return facts instead of sentences |
| [Staying responsive](#staying-responsive) | The three threads, and the timings that forced them |
| [Integrations](#integrations) | Calendar, weather, smart home, email — each with a demo backend |
| [Project layout](#project-layout) | Where everything lives |

**Safety and trust**
| | |
|---|---|
| [Asking before acting](#asking-before-acting) | The spoken confirmation gate, and what's refused outright |
| [JANET as an agent](#janet-as-an-agent) | Running code, editing its own source, and the containment around both |
| [Checking it still works](#checking-it-still-works) | The two check suites |

**Maintaining it**
| | |
|---|---|
| [Training the intent classifier](#training-the-intent-classifier) | Required for a fresh clone |
| [Principles](#principles) · [Target stack](#target-stack) | The rules the code is held to |

## How it works

In plain terms, one sentence at a time:

1. The microphone runs continuously, and **Silero VAD** watches for the moment
   speech starts and stops. That slice of audio is one *utterance*.
2. **Whisper** transcribes it locally.
3. Two cheap gates decide whether it was aimed at JANET at all. If either says
   no, JANET stays silent and nothing further runs.
4. If it passes, a **handler** does the actual work — sets the timer, reads the
   calendar, switches the light — and returns plain **facts**, not a sentence.
5. A **local language model** turns those facts, plus the conversation so far,
   into what you actually hear.
6. **espeak-ng** speaks it.

The whole path, with the file that owns each stage:

```
Mic ─► VAD ─► Whisper ─► [pending yes/no?] ─► Scorer ─► Classifier ─► Action ─► LLM ─► TTS
      Silero   (STT)       utils/confirm      (Layer 1)  (Layer 2)    handler   voice
                                                                        │         │ ↑
                                                                      facts ──────┘ │
                                                                    read-only tools ┘
                                                              (when the facts fall short)

 └── capture thread ──┘└──────────── main thread ────────────────────┘└ speech thread ┘
```

Those are three real threads, so **the microphone never stops** while JANET
thinks or talks — see [Staying responsive](#staying-responsive).

Nothing is spoken unless **both** gates agree the utterance is a real request *to JANET*:

1. **Scorer** (`intent/scorer.py`) — "Was this addressed to me?" Scores linguistic signals (question/command shape, "janet", keywords…) against a threshold. Cheap, runs first.
2. **Classifier** (`intent/classifier.py`) — "What do they want?" A fine-tuned DistilBERT (13 intents, ~97% val accuracy). Also vetoes anything it reads as `NONE` (not a real intent) or is unsure about.

If either gate says no, JANET stays silent — which matters a lot for an always-listening mic.

**With one exception** — the *"and tomorrow?"* case from the conversation at the
top. Both gates judge a single sentence with no idea what JANET just said, so a
short follow-up scores 0 and falls straight through them.

So before going quiet, JANET asks the LLM — the only part of it that actually has
the conversation — whether it was being talked to. If yes, the utterance is
answered and can use the tools below.

That check only runs when JANET spoke in the last 30 seconds *and the utterance
is short*, or the score was near the line — so a quiet room, or a background
video with no conversation in progress, still costs nothing at all. The length
rule matters more than it sounds: a reply to JANET is short (the longest real
one measured was 8 words), while media arrives in 30-word paragraphs, and one of
those was getting through.

It also verifies itself in the other direction. A bare question scores exactly
the threshold on question-shape alone, so *any* question in the room used to
clear the first gate — and a line of film dialogue ("Why are we going this
way?") got answered. When an utterance passes on question-shape alone and the
classifier can only call it "general", JANET now checks before replying rather
than after.

**It also decides where the utterance goes**, which turned out to matter more
than the rescue itself. Polite phrasings score badly on the classifier — "can
you delete lunch with Alex?" came out at 48% confidence, under the floor — and
sending those to the general-knowledge path meant JANET answered *"I can't
delete events for you"* about something it does perfectly well. Three different
capabilities got falsely refused that way in one sitting.

So the check reports which of two things it heard:

| | Goes to | Because |
|---|---|---|
| **a new request** — "can you turn it up to 60" | the real handler | it's something to *do*, whatever the classifier's confidence said |
| **a follow-up** — "and tomorrow?" | the LLM + its tools | it only makes sense in context; the calendar handler would read out events instead of the weather |

Ties go to *new request*: refusing something you can do is worse than acting on
the wrong thing.

One more thing it filters — **acknowledgements get no reply.** "Okay", "thanks",
"got it" are aimed at JANET but aren't asking for anything, and answering them
makes it sound needy.

The one thing that jumps the queue is an answer to a pending confirmation (see
[Asking before acting](#asking-before-acting)) — a bare "yes" carries no linguistic
signal at all, so the scorer would discard it as background chatter.

## Current capabilities

Every intent the classifier recognises now has a handler:

| Intent | What you can say |
|---|---|
| **TIME / DATE** | "what time is it", "what's the date" |
| **TIMER** | "set a timer for 5 minutes", "set a pasta timer for 12 minutes", "how much time is left", "cancel the timer" |
| **ALARM** | "set an alarm for 7 AM", "every weekday at 8", "set an alarm for 8 on Wednesday", "cancel the alarm" |
| **REMINDER** | "remind me to call mom in ten minutes", "remind me to take the pills at 5", "what are my reminders" |
| **CALC** | "what's 25% of 52", "twenty times three", "the derivative of x squared plus three x", "solve x squared equals 4", "what's 5 choose 2" |
| **CALENDAR** | "what's on my calendar today", "what's my next meeting", "am I free tomorrow", "schedule a dentist appointment tomorrow at 3", "remove lunch with Alex" |
| **WEATHER** | "what's the weather", "will it rain tomorrow", "what's the weather in London" |
| **SMART_HOME** | "turn on the living room lights", "switch off the fan", "are the kitchen lights on" |
| **EMAIL** | "do I have any new email", "who emailed me", "read my last email", "reply saying I'll be there" |
| **SYSTEM** | "turn the volume up", "set the volume to 40", "mute", "say that again", "what can you do" |
| **GENERAL** | anything else → answered by the local LLM |

Anything JANET decides wasn't addressed to it gets **silence**, not a reply.

## Running it

*Installing it is [📦 Install](#-install) above, or
[instructions.md](instructions.md) in full. This is what happens once it's set
up.*

```bash
source venv/bin/activate
cd src && python main.py          # from src/, as your normal user — see below
```

Two rules about *how* you start it, both of which fail silently rather than
loudly if you get them wrong:

- **Not `sudo`.** A per-user PipeWire mic is unreachable as root, so capture
  comes out empty with no error at all.
- **From inside `src/`.** Modules import by top-level package
  (`from intent.scorer import score`), so `src/` has to be the working directory.

**You also need a local LLM running**, since it is JANET's voice. Either works:

```bash
# llama.cpp (default) — fastest if you give it the whole GPU
llama-server -m /path/to/qwen3-14b-instruct-q4_k_m.gguf -ngl 999 -c 8192 --port 8081

# or Ollama, with JANET_LLM_URL=http://127.0.0.1:11434/v1 in your .env
ollama pull qwen3:14b && ollama serve
```

Don't run both at once on one GPU: the second copy of the weights gets squeezed
onto the CPU and every reply becomes 3-5x slower. Without any model JANET still
works, but falls back to plain canned replies and says so.

Nothing in `.env` is required: the time, timers, alarms, reminders, maths, system
control and the local LLM all work with no configuration at all.

**Then just talk.** There's nothing to press and no wake word to say. Silero VAD
detects when you start and stop speaking (a short pre-roll buffer keeps your
first word), and the console shows the decision on every utterance:

```
🗣  You said: Janet, what time is it?
🧠 Intent: TIME (91%)
🛡  Score: 115 (janet +50, question +40, keyword +25) → addressed
⚙️  Reply: It's 8:33 PM.
💭 Why: The time is given in the facts.
```

When JANET stays quiet it says why, which is most of what you need to debug it —
`🛡 … → ignored` is a gate refusing outright, `🤔 … → not for me` is the LLM
agreeing after a second look, and `🤔 … → rescued` is the LLM overruling them
both. `Ctrl-C` to quit.

Mishearings are the usual cause of "why didn't it answer me". `JANET_DEBUG_AUDIO=1`
saves each utterance as a WAV with its duration, which tells a capture problem
apart from a model one — [instructions.md §11](instructions.md#11-troubleshooting)
walks through the rest.

## The LLM does the talking

Handlers don't speak any more. A handler runs, returns a short factual string,
and a **local LLM** turns those facts plus the conversation into what you
actually hear — so JANET has one voice everywhere and knows what was just said:

```
you:   Janet, set a timer for 5 minutes
JANET: I've set a timer for 5 minutes.          (fact: "Timer set for 5 minutes.")
you:   Janet, how much time is left?
JANET: There's 4 minutes and 58 seconds left.
```

**It can also go and look things up.** If what the action returned doesn't answer
the question, JANET says something casual and fetches the rest itself:

```
you:   Janet, what's the weather?
JANET: It's 22 degrees and partly cloudy in Pune.
you:   Janet, what about in Delhi?
JANET: Let me check.                        (calls get_weather(city="Delhi"))
JANET: It's 22.4 degrees and partly cloudy in Delhi.
```

Those lookups are **read-only** — weather, calendar, time, email, device state.

**And it can act, too.** If you ask for something to be done, the model runs the
action itself rather than depending on the intent classifier to have understood
you. That matters more than it sounds: polite phrasings score badly on the
classifier ("can you delete lunch with Alex?" came out at 48% confidence), and
JANET used to answer *"I can't delete events for you"* about things it does
perfectly well.

JANET only offers itself actions when nothing has run yet — once a handler has
answered, the action for that sentence has already happened. That also keeps the
list of choices short, which turns out to matter: a longer list made it
occasionally reach for the wrong one.

**Acting never skips a confirmation.** The gate lives inside the handler, not in
how the handler was reached — so creating an event or sending a reply still
states itself back and waits for a yes, whichever route got there. Read-only
lookups and world-changing actions are kept in separate files so the code that
can *change* anything is one short list you can read in a minute. An action can
run at most once per request, and lookups are capped at two.

### It can't tell you it did something it didn't

The rule is simple and mechanical: **the facts a handler returns are the only
evidence an action ran.** If a reply claims something was set, added, sent or
switched and no facts back it up, the claim is false by construction — so JANET
says it couldn't do it instead.

This exists because of a real failure, caught by checking JANET's word against
the system it claimed to have changed:

```
you:   Can you turn it down to 55 again?
JANET: Okay, I've set the volume to 55%.
pactl: 80%                                   ← nothing had run
```

For an assistant with no screen, *silently didn't do it but said it did* is
worse than any error message — you walk away believing the light is off. There
are three layers against it now (the turn states outright that nothing ran, a
prompt rule, and a check on the finished sentence), because the first fix wasn't
enough: told not to claim the *action*, the model asserted the *result state*
instead. See `ai_core/claims.py`.

One lesson from building that prompt, since it cost three separate bugs: **a
negative rule aimed at an edge case gets applied to the main case.** Telling the
model "never announce what you're about to say" — and quoting the bad phrasing —
made it do that *more*. Telling it "never produce a number the calculator
declined" made it refuse to read out numbers the calculator had happily
produced. Both were fixed by saying what to do first and putting the exception
second.

It also records *why* it said something (a one-line reasoning), and can decide a
question needs real thought — saying something casual first so the pause isn't
dead air:

```
you:   Janet, if a train leaves at 3:40pm at 80km/h and another at 4:10pm at 110km/h, when does it catch up?
JANET: hang on, let me think about that one
JANET: The first train has a 30-minute head start, covering 40 km. The second
       gains 30 km/h, so it takes 1 hour 20 minutes. They meet at 5:30 pm.
```

That deeper mode is off by default (`JANET_DEEP_THINKING=1`) because it takes
~11–20s — and while JANET keeps listening throughout, it can't answer you until
it finishes.

Everything — query, facts, reasoning, thinking, reply — is written to a
gitignored transcript in `data/transcripts/`.

**If the model is unavailable**, JANET falls back to the handlers' own plain
strings and tells you once that its replies will be basic for now, rather than
going silent.

The backend is just a URL (`JANET_LLM_URL`): a local **llama.cpp** server by
default, or **Ollama**. Everything stays on-device either way.

JANET also keeps a **short-term conversation memory** (`utils/history.py`) — the last few addressed exchanges are fed back to the LLM, so follow-ups work: *"what's the capital of France?"* → *"Paris"*, then *"what about Germany?"* → *"Berlin."* It's in-memory and resets on restart.

**ALARM** sets one-shot, specific-day, and recurring alarms by voice and can cancel them — *"set an alarm for 7 AM"*, *"set an alarm for 8 on Wednesday"*, *"every weekday at 8"*, *"cancel the alarm"*. A bare time resolves to the soonest future occurrence (parsing in `intent/timeparse.py`, scheduling in `actions/alarm_action.py`). Alarms are saved to disk and re-armed on restart — see [Asking before acting](#asking-before-acting) for what happens to one that came due while JANET was off.

**CALC** works on a principle worth stating plainly: **the language model translates, and a real maths library computes.** Models are fluent and unreliable at arithmetic; SymPy is the reverse. So the model only turns your sentence into an expression — it never calculates, and it never writes executable code (the operation is picked from a fixed list, and expressions are vetted as syntax before anything can run them).

That means JANET handles far more than sums: *"the derivative of x squared plus three x"* → `2x + 3`, *"solve x squared equals 4"* → `x = -2, 2`, *"the integral of 2x"*, *"5 choose 2"* → 10, limits, factoring and expanding. Ordinary arithmetic still takes a fast deterministic path with no model call at all — *"twenty times three"* → 60, *"two to the power of ten"* → 1024, *"25% of 52"* → 13.

**And it declines rather than guessing.** That fast parser answers only when it can account for every number you said. Live testing caught Whisper turning *"25% **of** 52"* into *"25% **to** 52"*, which the old code parsed as just `25%` and answered **0.25** — for a question whose answer is 13. Dropping one of your numbers now hands the sentence to the solver instead of inventing an answer. Number-word parsing lives in `intent/numwords.py`; the solver in `ai_core/mathsolve.py`.

## Staying responsive

Every stage was timed before any of it was threaded, and the result was not what
you'd guess:

| Stage | Warm time |
|---|---|
| Whisper speech-to-text | **0.10s** |
| Intent classifier | **0.00s** |
| The language model | 1.4–2.3s |
| **Speaking the answer** | **8.54s** |

Speaking isn't a detail at the end of the pipeline — it's about 80% of it. And
JANET used to sit inside it with the microphone stopped, so anything you said
while it was talking was never even recorded.

Capture, thinking and speaking now run on separate threads. The mic keeps
running throughout, models load at startup instead of during your first sentence
(3.6s that used to land exactly when JANET felt broken), and everything that
talks shares one speech queue — so an alarm going off mid-answer waits its turn
instead of playing over the top, which is what used to happen.

**JANET still won't listen while it speaks**, on purpose: without echo
cancellation it would transcribe its own voice, answer itself, and do it again.
If you want barge-in, load the echo canceller and tell JANET the mic is safe:

```bash
pactl load-module module-echo-cancel   # not persistent across reboots
JANET_BARGE_IN=1 python main.py
```

## Checking it still works

```bash
export JANET_CALENDAR=demo JANET_WEATHER=demo JANET_SMART_HOME=demo JANET_EMAIL=demo
./venv/bin/python tools/smoke.py        # 30 checks, ~1 min — the fast gate
./venv/bin/python tools/full_check.py   # 69 checks — every intent, gate and guard
```

`full_check.py` covers all twelve intents, the maths solver and its blocked
attack surface, the acting path, the honesty guard, ambient rejection, the
confirmation vocabulary, calendar writes end to end, persistence across a
restart, the speech thread, agent containment, and what happens when the model
is unreachable.

## Integrations

Each integration hides behind a small protocol (`integrations/*_source.py`), so the
handler never knows which backend it's talking to — and **each one ships a fake
"demo" backend** with believable sample data. That means you can try JANET's
calendar, weather, smart home and email with **no accounts and no credentials**,
and it's how the whole thing is tested.

| | Real backend | Demo |
|---|---|---|
| **Calendar** | CalDAV (self-hostable → fully local) or Google Calendar | `JANET_CALENDAR=demo` |
| **Weather** | [Open-Meteo](https://open-meteo.com) — free, **no API key** | `JANET_WEATHER=demo` |
| **Smart home** | Home Assistant over your own LAN | `JANET_SMART_HOME=demo` |
| **Email** | IMAP (read) + SMTP (reply) | `JANET_EMAIL=demo` |

Copy [`.env.example`](.env.example) to `.env` and fill in only what you use. Try it
with no setup at all:

```bash
cd src && JANET_CALENDAR=demo JANET_WEATHER=demo JANET_SMART_HOME=demo JANET_EMAIL=demo python main.py
```

## Asking before acting

Whisper mishears — a lot. That's harmless for a timer and dangerous for anything
that writes to the outside world, so actions that are hard to undo go through a
spoken confirmation (`utils/confirm.py`): JANET states the whole thing back and
does it only if your **next** utterance is a yes.

```
you:   Janet, schedule a physio appointment tomorrow at 4 pm
JANET: Shall I add Physio appointment tomorrow at 4 PM?
you:   yes
JANET: Added Physio appointment to your calendar.
```

Saying anything other than yes/no cancels it and is handled as a fresh command, and
the request expires after a minute so a stray "yeah" later can't trigger it.

**And what it promises, it keeps.** Alarms, timers and reminders are saved to
disk and re-armed when JANET starts. They used to be in-memory timers, so a
restart threw them away silently — JANET said the alarm was set, and it was,
right up until it wasn't. Missed events aren't all treated alike: a recurring
alarm re-arms, a one-shot alarm or timer whose moment passed is dropped (saying
"it's 7 AM" at half past nine would be false), but a missed **reminder** is
still spoken, because "you asked me to remind you to call mom" is just as useful
late.

**Yes doesn't have to be tidy.** People answer with "oh yeah", "um, sure", "well,
go ahead" — and decline by reassuring you ("no it's okay", "that's fine") rather
than refusing. Conversational padding is stripped before matching, so those all
land. Only meaningless fillers are removed, so nothing can flip an answer.

And because JANET now knows when it has asked you something, **an answer to its
own question is never mistaken for background chatter** — which it used to be,
every single time.

Some things are refused outright rather than confirmed, because no wake word means
one misheard sentence shouldn't be able to cause real damage:

- **No power commands** — shutdown, reboot, suspend and logout aren't implemented at all.
- **No locks, garage doors or alarm panels** — smart-home control is limited to lights, switches and fans.
- **Reading email never marks it read** (IMAP `BODY.PEEK`), and JANET can only *reply* to a message, never compose to an arbitrary address.

## JANET as an agent

JANET can run code, work with files, and **propose changes to its own source** —
all behind permission, and all contained:

| What | Where it runs | How it's approved |
|---|---|---|
| Run code / commands | **bubblewrap sandbox** — no home, no network, no credentials | spoken "yes" |
| Create/edit/delete files | a workspace directory, not your disk | spoken "yes" |
| Look at the real machine | a read-only allowlist (`git status`, `nvidia-smi`, …) | spoken "yes" |
| **Edit its own code** | a git branch, tested, never `master` | **you read the diff** |

The last row is the important one. You can hear a summary; you cannot hear a
diff — so a spoken "yes" only authorises JANET to *write a proposal*:

```
JANET: Shall I make the timer understand "how much longer"?
you:   yes
JANET: I've written the design to docs/janet_changes/… — have a read.
you:   go ahead
JANET: Done, on branch janet/20260729-…  Syntax clean, 30/30 smoke passed.
       Review the diff and merge it if you're happy.
```

JANET never merges and never restarts itself. Some files it will simply refuse
to touch — the confirmation gate, the addressing gate, the sandbox, and this
list — because *"remove the confirmation gate"* is otherwise a perfectly valid
request. Change those by hand or not at all.

It also reports what a change **removed**, not just whether it passed:

```
Syntax clean, 30/30 smoke passed.
removed: src/intent/timerparse.py: lost 1 docstring(s)
```

That check exists because the model really does this. Asked for a small change,
it twice rewrote a whole file and deleted its module docstring — passing every
test both times, because the behaviour was identical. Telling it not to didn't
help, so JANET now diffs before and after and says what vanished. Tests prove a
change didn't *break* anything; only reading the diff shows what it *took away*.

Those are the same suites from [Checking it still works](#checking-it-still-works)
— JANET runs them against its own proposed diff before handing it to you.

## Training the intent classifier

**A fresh clone has no trained model** — `data/models/` is gitignored, so this is
a required setup step, not an optional one. JANET won't start without it (it
fails with a message telling you to run exactly this):

```bash
cd src && python -m intent.train
```

This reads the labelled dataset in `data/text/{train,val}/` — 2,005 training and
504 validation utterances across 13 labels, all committed — fine-tunes
`distilbert-base-uncased`, prints per-class metrics + a confusion matrix, and
saves the model to `data/models/intent-distilbert/`.

Measured on an RTX 5060 Ti: **16 seconds, 97.2% validation accuracy** (macro F1
0.974). Budget 10–20 minutes on a CPU — a GPU helps but genuinely isn't required.

Validate the dataset's format with `python data/text/validate.py`. And since the
data is just text files — one utterance per line, one file per label — **adding
the way you actually talk and retraining is the cheapest way to make JANET
understand you better.**

## Project layout

```
src/
  main.py            always-listening loop
  audio/             frames() source, Silero VAD, ring buffer, Whisper STT, TTS
  ai_core/           llm (the one model client) · responder (JANET's voice) · tools (read-only lookups)
                     sandbox (bubblewrap) · workspace (confined files) · host (read-only allowlist)
                     selfmod (JANET editing its own code, reviewed)
                     addressing (rescues follow-ups) · transcript
  intent/            normalize · scorer (Layer 1) · classifier + train/dataset (Layer 2) · dispatch
                     parsers: timeparse (alarms) · timerparse · remindparse · dateparse · eventparse · numwords
  actions/           one handler per intent (time, date, timer, alarm, reminder, calc,
                     calendar, weather, smart_home, email, system, general)
  integrations/      calendar (CalDAV · Google · demo), weather (Open-Meteo · demo),
                     smart home (Home Assistant · demo), email (IMAP/SMTP · demo)
  utils/             context, conversation memory (history.py), confirm gate, helpers
data/
  text/              intent dataset (train/val), labels.txt, validate.py
  models/            trained model — GITIGNORED, you train it (instructions.md §5)
  state/             saved alarms/timers/reminders (gitignored, made at runtime)
  transcripts/       per-day JSONL of every turn (gitignored)
tools/
  smoke.py           30 checks over the real pipeline — the fast gate
  full_check.py      69 checks — every intent, gate and guard
instructions.md      full setup guide
.env.example         every setting, documented inline
```

**A fresh clone is missing a few things by design** — the trained model, `.env`,
your saved alarms, and the transcripts. Everything needed to rebuild them is
committed; [instructions.md §10](instructions.md#10-what-your-clone-does-not-contain)
lists exactly what's absent and what to do about each.

## Principles

- **Local-first.** No cloud dependency for any core function — that's the whole point.
- **Understandable over clever.** Small, reviewable changes; the owner wants to explain every line.
- **Human-in-the-loop.** Nothing acts on you unless both gates agree it was meant for JANET.

## Target stack

Whisper (STT, `small`, English-forced and vocabulary-primed) · Silero VAD · DistilBERT intent classifier · a multi-signal addressing scorer · a local LLM as the voice (Qwen3 14B via llama.cpp or Ollama) · Piper TTS (planned). Core speech and reasoning are entirely on-device; only the optional calendar/weather/smart-home/email integrations touch the network, and each has a local or self-hostable option.
