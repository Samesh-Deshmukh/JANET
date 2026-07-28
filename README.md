# JANET

**Just Another Neural Execution Tool** — a fully-local, privacy-first, wake-word-free voice assistant.

Everything runs on-device. No cloud calls for core functions; no wake word to press or say — JANET listens continuously and works out for itself when you're talking to it.

> **Status: working prototype.** The audio pipeline, the intent brain (addressing scorer + trained classifier), and **every intent handler** are built: time, date, timers, alarms, reminders, maths, calendar, weather, smart home, email, system control, and open-ended questions. Since the latest change, **a local LLM does all the talking** — handlers produce facts, the model turns them into speech. This is also a personal learning project — the code favours being understandable over clever.

## The pipeline

```
Mic ─► VAD ─► Whisper ─► [pending yes/no?] ─► Scorer ─► Classifier ─► Action ─► LLM ─► TTS
      Silero   (STT)       utils/confirm      (Layer 1)  (Layer 2)    handler   voice
                                                                        │         │
                                                                      facts ──────┘
```

Nothing is spoken unless **both** gates agree the utterance is a real request *to JANET*:

1. **Scorer** (`intent/scorer.py`) — "Was this addressed to me?" Scores linguistic signals (question/command shape, "janet", keywords…) against a threshold. Cheap, runs first.
2. **Classifier** (`intent/classifier.py`) — "What do they want?" A fine-tuned DistilBERT (13 intents, ~97% val accuracy). Also vetoes anything it reads as `NONE` (not a real intent) or is unsure about.

If either gate says no, JANET stays silent — which matters a lot for an always-listening mic.

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
| **CALC** | "what's 25% of 52", "twenty times three", "two to the power of ten", "square root of 144" |
| **CALENDAR** | "what's on my calendar today", "what's my next meeting", "am I free tomorrow", "schedule a dentist appointment tomorrow at 3" |
| **WEATHER** | "what's the weather", "will it rain tomorrow", "what's the weather in London" |
| **SMART_HOME** | "turn on the living room lights", "switch off the fan", "are the kitchen lights on" |
| **EMAIL** | "do I have any new email", "who emailed me", "read my last email", "reply saying I'll be there" |
| **SYSTEM** | "turn the volume up", "set the volume to 40", "mute", "say that again", "what can you do" |
| **GENERAL** | anything else → answered by the local LLM |

Anything JANET decides wasn't addressed to it gets **silence**, not a reply.

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
~11–20s and JANET is single-threaded, so the mic is deaf while it thinks.

Everything — query, facts, reasoning, thinking, reply — is written to a
gitignored transcript in `data/transcripts/`.

**If the model is unavailable**, JANET falls back to the handlers' own plain
strings and tells you once that its replies will be basic for now, rather than
going silent.

The backend is just a URL (`JANET_LLM_URL`): a local **llama.cpp** server by
default, or **Ollama**. Everything stays on-device either way.

JANET also keeps a **short-term conversation memory** (`utils/history.py`) — the last few addressed exchanges are fed back to the LLM, so follow-ups work: *"what's the capital of France?"* → *"Paris"*, then *"what about Germany?"* → *"Berlin."* It's in-memory and resets on restart.

**ALARM** sets one-shot, specific-day, and recurring alarms by voice and can cancel them — *"set an alarm for 7 AM"*, *"set an alarm for 8 on Wednesday"*, *"every weekday at 8"*, *"cancel the alarm"*. A bare time resolves to the soonest future occurrence (parsing in `intent/timeparse.py`, scheduling in `actions/alarm_action.py`). Alarms are in-memory and reset on restart.

**CALC** does arithmetic plus exponents, modulo, square root, percentages, and **spoken number words** — *"twenty times three"* → 60, *"two to the power of ten"* → 1024, *"square root of 144"* → 12, *"twenty percent of fifty"* → 10. It requires a real operator (so a misheard number isn't answered as a "calculation"), rounds non-integer results, and refuses divide-by-zero. Number-word parsing lives in `intent/numwords.py`.

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

Some things are refused outright rather than confirmed, because no wake word means
one misheard sentence shouldn't be able to cause real damage:

- **No power commands** — shutdown, reboot, suspend and logout aren't implemented at all.
- **No locks, garage doors or alarm panels** — smart-home control is limited to lights, switches and fans.
- **Reading email never marks it read** (IMAP `BODY.PEEK`), and JANET can only *reply* to a message, never compose to an arbitrary address.

## Getting started

Requires audio hardware (mic + speaker) and **Python 3.11**. Run as your **normal user** (not `sudo` — a per-user PipeWire mic is unreachable as root).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # optional — see below
cd src && python main.py          # runs from src/ — see the import note below
```

**You need a local LLM running**, since it is JANET's voice. Either works:

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

Then just talk. Silero VAD detects when you start and stop speaking (a short pre-roll buffer keeps your first word). Each utterance is transcribed, run through the two gates, and — if it's for JANET — acted on and spoken. The console shows the decision on every utterance:

```
🗣  You said: what time is it
🛡  Score: 65 (question +40, keyword +25) → addressed
🧠 Intent: TIME (89%)
⚙️  Reply: The time is 07:51 PM
```

`Ctrl-C` to quit.

> **Import convention:** modules import by top-level package (`from intent.scorer import score`), so `src/` must be the working directory — run from inside `src/`, not the repo root.

## Training the intent classifier

The classifier is fine-tuned locally (a GPU helps but isn't required):

```bash
cd src && python -m intent.train
```

This reads the labelled dataset in `data/text/{train,val}/`, fine-tunes `distilbert-base-uncased`, prints per-class metrics + a confusion matrix, and saves the model to `data/models/intent-distilbert/` (gitignored). Validate the dataset's format with `python data/text/validate.py`.

## Project layout

```
src/
  main.py            always-listening loop
  audio/             frames() source, Silero VAD, ring buffer, Whisper STT, TTS
  ai_core/           llm (the one model client) · responder (JANET's voice) · transcript
  intent/            normalize · scorer (Layer 1) · classifier + train/dataset (Layer 2) · dispatch
                     parsers: timeparse (alarms) · timerparse · remindparse · dateparse · eventparse · numwords
  actions/           one handler per intent (time, date, timer, alarm, reminder, calc,
                     calendar, weather, smart_home, email, system, general)
  integrations/      calendar (CalDAV · Google · demo), weather (Open-Meteo · demo),
                     smart home (Home Assistant · demo), email (IMAP/SMTP · demo)
  utils/             context, conversation memory (history.py), confirm gate, helpers
data/
  text/              intent dataset (train/val), labels.txt, validate.py
  models/            trained model (gitignored)
  transcripts/       per-day JSONL of every turn (gitignored)
```

## Principles

- **Local-first.** No cloud dependency for any core function — that's the whole point.
- **Understandable over clever.** Small, reviewable changes; the owner wants to explain every line.
- **Human-in-the-loop.** Nothing acts on you unless both gates agree it was meant for JANET.

## Target stack

Whisper (STT) · Silero VAD · DistilBERT intent classifier · a multi-signal addressing scorer · a local LLM as the voice (Qwen3 14B via llama.cpp or Ollama) · Piper TTS (planned). Core speech and reasoning are entirely on-device; only the optional calendar/weather/smart-home/email integrations touch the network, and each has a local or self-hostable option.
