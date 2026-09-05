# Installing JANET

A step-by-step setup guide for a **fresh clone on a machine that isn't the
author's**. If you just want to know what JANET *is*, read
[README.md](README.md) first — this file assumes you've decided to run it.

Follow the steps in order. You don't have to train anything: the intent
classifier's config and tokenizer are committed, and JANET downloads its 256 MB
weights file automatically the first time it starts. Step 5 explains it.

---

## Contents

1. [Before you start — is this going to work?](#1-before-you-start--is-this-going-to-work)
2. [Install the system packages](#2-install-the-system-packages)
3. [Clone the repo](#3-clone-the-repo)
4. [Create the virtualenv and install Python packages](#4-create-the-virtualenv-and-install-python-packages)
5. [The intent classifier (downloads itself)](#5-the-intent-classifier-downloads-itself)
6. [Run a local LLM — JANET's voice](#6-run-a-local-llm--janets-voice)
7. [Configure `.env` (optional)](#7-configure-env-optional)
8. [First run](#8-first-run)
9. [Verify the install](#9-verify-the-install)
10. [What your clone does *not* contain](#10-what-your-clone-does-not-contain)
11. [Troubleshooting](#11-troubleshooting)
12. [Disk usage and uninstalling](#12-disk-usage-and-uninstalling)
13. [macOS, Windows, and everything else](#13-macos-windows-and-everything-else)

---

## 1. Before you start — is this going to work?

JANET was built and is run on **Arch Linux with PipeWire**. Nothing in it is
Arch-specific, and only two files are Linux-specific — but it's better to know
which parts apply to you now than after a 4 GB download.

### Requirements

| | Needed | Notes |
|---|---|---|
| **OS** | Linux with PipeWire or PulseAudio, **or** WSL2 on Windows 11 | Those are the paths that work with no code changes. macOS and native Windows need two small files ported — [§13](#13-macos-windows-and-everything-else) walks through exactly which, and how. |
| **Python** | **3.11** | What every pin in `requirements.txt` is tested against. 3.12/3.13 may work; you're on your own for wheel availability. |
| **Microphone** | Required | JANET is always-listening. There is no text-only mode. |
| **Speakers** | Required | Every reply is spoken; nothing is printed *instead*. |
| **Disk** | **~10 GB** | The virtualenv alone is ~7.9 GB, and the CUDA PyTorch wheel is most of that — a **CPU or macOS build is far smaller**, closer to 3 GB total. Plus ~3.2 GB of models (the addressing gate is ~1.7 GB of that), plus your LLM. |
| **RAM** | 8 GB+ | 16 GB if you want a comfortable LLM alongside. |
| **GPU** | Strongly recommended | Not required for JANET itself — see below. |
| **Internet** | First run only | To download the model weights. After that JANET runs fully offline. |

### About the GPU

JANET has four models, and only one of them really cares:

| Model | On CPU | On GPU |
|---|---|---|
| Silero VAD | fine (it's 2 MB) | fine |
| Whisper `small` | usable, a few seconds per utterance — drop to `tiny` if it drags | ~0.1 s |
| DistilBERT intent | fine (~0 s per utterance either way) | fine |
| NLI addressing gate | usable (a fraction of a second per utterance) | ~0.01 s, 848 MB VRAM |
| **The LLM (JANET's voice)** | **painful — expect 30 s+ per reply** | 1–3 s |

**The addressing gate downloads ~1.7 GB on first run.** It is
`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`, pulled from Hugging Face and
cached in `~/.cache/huggingface`. It needs no extra pip package — the
checkpoint ships a fast tokenizer, so **`sentencepiece` is not required**. If
VRAM is tight, put `JANET_NLI_MODEL=MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
in your `.env`: 379 MB instead of 848 MB, and measurably worse at telling your
own conversation apart from a request.

So: **no GPU means you should use a much smaller LLM** (a 3B, not a 14B), or
accept that JANET pauses for a long time before every sentence. Everything else
runs acceptably on a CPU.

"GPU" here means anything with real acceleration — an NVIDIA card via CUDA, or
**Apple Silicon via Metal**. An M-series Mac is a perfectly good JANET machine:
PyTorch uses MPS for Whisper and the classifier, and llama.cpp's Metal backend
runs the LLM at comparable speed to a mid-range NVIDIA card.

### Not on Linux?

Read [§13](#13-macos-windows-and-everything-else) **before** you start
installing. Short version:

| | Verdict |
|---|---|
| **Linux** | Works as documented. |
| **Windows 11 via WSL2** | Works as documented — WSLg gives you PulseAudio, mic included. Easiest non-Linux route by a wide margin. |
| **macOS** | Two files need porting (~30 lines). Everything else already works, and macOS has better built-in tools for both of them. |
| **Windows, natively** | Same two files, plus `pycaw` for volume. `winsound` is in the standard library. |

The rest of this guide is written for Linux. Where a step differs, the relevant
section says so.

---

## 2. Install the system packages

JANET shells out to a handful of ordinary command-line tools. Install them with
your platform's package manager — Linux distros first, then
[WSL2](#windows-11-via-wsl2-the-recommended-windows-route),
[macOS](#macos) and [native Windows](#windows-natively) below.

| Tool | Why JANET needs it | Required? |
|---|---|---|
| `paplay`, `pactl` | plays JANET's speech, and reads/sets volume (`audio/tts.py`, `utils/volume.py`) | **yes** |
| `espeak-ng` | the **fallback** voice only. JANET speaks with Piper, which pip installs and which bundles its own phonemiser — so espeak-ng is no longer the voice. Keep it: if the Piper model can't load, this is what stops JANET going silent. | recommended |
| PortAudio | microphone capture, via PyAudio (`audio/audio.py`) | **yes** |
| C compiler + Python headers | only if pip has to build PyAudio from source | if no wheel |
| `git` | cloning | **yes** |
| `ffmpeg` | Whisper uses it to read audio *files*. The live mic path hands Whisper a NumPy array and never touches ffmpeg — you only need this for the offline debug harnesses. | recommended |
| `bubblewrap` (`bwrap`) | the sandbox JANET runs code in (`ai_core/sandbox.py`). Without it JANET refuses to run code rather than running it unsandboxed — which is the correct behaviour, so this is genuinely optional. | optional |

### Arch / Manjaro / EndeavourOS

```bash
sudo pacman -S --needed espeak-ng libpulse portaudio ffmpeg bubblewrap git base-devel
```

### Debian / Ubuntu / Pop!_OS / Mint

```bash
sudo apt update
sudo apt install espeak-ng pulseaudio-utils portaudio19-dev ffmpeg bubblewrap git \
                 build-essential python3.11 python3.11-venv python3.11-dev
```

> On Ubuntu 22.04 and later, `python3.11` may need the deadsnakes PPA:
> `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update`

### Fedora

```bash
sudo dnf install espeak-ng pulseaudio-utils portaudio-devel ffmpeg-free bubblewrap git \
                 gcc python3.11 python3.11-devel
```

### openSUSE

```bash
sudo zypper install espeak-ng pulseaudio-utils portaudio-devel ffmpeg bubblewrap git \
                    gcc python311 python311-devel
```

> **`pulseaudio-utils` on a PipeWire system is correct**, not a mistake. `pactl`
> and `paplay` are the PulseAudio *client* tools, and PipeWire's compatibility
> layer (`pipewire-pulse`) answers them. Most modern distros ship this already.

### Windows 11, via WSL2 (the recommended Windows route)

WSLg gives a WSL2 distro a working PulseAudio server — speakers *and* microphone
— so JANET runs unmodified. From PowerShell:

```powershell
wsl --install -d Ubuntu
wsl --update          # WSLg ships with recent WSL; make sure you're current
```

Then open the Ubuntu shell and follow the **Debian/Ubuntu** instructions above
and the rest of this guide exactly as written. Check audio reached the VM before
going further:

```bash
pactl info                    # should print a PulseAudio server, not an error
pactl list sources short      # should list a microphone (RDP source)
```

Mic passthrough quality varies with your Windows audio setup — if `pactl list
sources short` is empty, `wsl --shutdown` and reopen; that fixes it more often
than anything else.

> CUDA works through WSL2 with a current Windows NVIDIA driver, so
> [§4a](#4a-you-have-an-nvidia-gpu) applies unchanged. Don't install a driver
> *inside* the distro.

### macOS

```bash
brew install python@3.11 portaudio espeak-ng ffmpeg
```

Then read [§13](#13-macos-windows-and-everything-else) — `paplay` and `pactl`
don't exist on macOS, so `audio/tts.py` and `utils/volume.py` need their
replacements before JANET will speak or change volume. Note also that **Terminal
needs microphone permission**: System Settings → Privacy & Security →
Microphone. Without it, capture is silent and nothing warns you.

### Windows, natively

Install [Python 3.11](https://www.python.org/downloads/) (tick "Add to PATH"),
[Git](https://git-scm.com/download/win), and the
[espeak-ng MSI](https://github.com/espeak-ng/espeak-ng/releases). PyAudio and
PyTorch both ship Windows wheels, so [§4a](#4a-you-have-an-nvidia-gpu) works
as-is with an NVIDIA card. Then see
[§13](#13-macos-windows-and-everything-else) for the two files to port.

### Check they're all there (Linux / WSL2)

```bash
for t in espeak-ng paplay pactl git ffmpeg bwrap; do
  printf '%-10s %s\n' "$t" "$(command -v $t || echo 'MISSING')"
done
```

`ffmpeg`, `bwrap` and `espeak-ng` may say MISSING without breaking the core
assistant — without `bwrap` JANET just declines to run code, which is the safe
answer, and without `espeak-ng` it loses only its fallback voice. `paplay`,
`pactl` and `git` missing will stop it working.

### Test your speaker right now

Before installing 8 GB of Python, confirm the thing JANET speaks through works.
This is the single highest-value 10 seconds in the whole guide — a silent
speaker looks identical to a broken assistant once it's buried in the pipeline.

(This tests `paplay`, which is what JANET actually plays through. It uses
espeak-ng to make the sound only because it needs *some* audio and espeak-ng is
already installed at this point — Piper isn't, until pip runs in §4.)

```bash
# Linux / WSL2
espeak-ng "Janet installation test" -w /tmp/janet-test.wav && paplay /tmp/janet-test.wav

# macOS
say "Janet installation test"

# Windows (PowerShell)
espeak-ng "Janet installation test" -w $env:TEMP\janet-test.wav
(New-Object Media.SoundPlayer "$env:TEMP\janet-test.wav").PlaySync()
```

If you didn't hear that, fix it before going any further.

---

## 3. Clone the repo

```bash
git clone https://github.com/Samesh-Deshmukh/JANET.git
cd JANET
```

---

## 4. Create the virtualenv and install Python packages

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows (native): venv\Scripts\activate
pip install --upgrade pip
```

> Every later command in this guide assumes the venv is **active**. If you open a
> new terminal, re-run the activate line first — a `ModuleNotFoundError` for
> `torch` or `whisper` is almost always this.

Now install the dependencies — **but read the right subsection first**, because
`requirements.txt` is pinned to a CUDA build of PyTorch.

### 4a. You have an NVIDIA GPU

Just install it. `requirements.txt` already points pip at the CUDA 12.8 wheel
index, which covers everything from Turing (RTX 20-series) through Blackwell
(RTX 50-series):

```bash
pip install -r requirements.txt
```

Then confirm PyTorch can actually see the card:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want: 2.11.0+cu128 True
```

If it prints `False`, your NVIDIA *driver* is too old for CUDA 12.8. Either
update the driver or follow 4b and run on the CPU.

### 4b. You have no GPU, an AMD/Intel GPU, or you're on macOS

The pinned `torch==2.11.0+cu128` only exists on PyTorch's CUDA index, so a plain
install will fail or drag in ~3 GB of CUDA libraries you can't use. Install
PyTorch yourself first, then everything else:

```bash
# CPU-only build (also the right choice for Intel integrated graphics)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# then the rest, with the CUDA pin and its index stripped out
grep -v '^torch==\|^--extra-index-url' requirements.txt > /tmp/requirements-nocuda.txt
pip install -r /tmp/requirements-nocuda.txt
```

On **macOS**, drop the `--index-url` entirely — `pip install torch` gives you the
standard wheel, which on Apple Silicon includes **MPS** (Metal) acceleration.
Whisper and the classifier will use the GPU without any further configuration.

For **AMD ROCm**, swap the first command for the ROCm index shown on
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) —
the ROCm version moves faster than this document does, so take it from there
rather than from here.

Everything in JANET works on a CPU build. Only speed changes.

> On Windows, the `grep` line above won't work. Open `requirements.txt` in an
> editor, delete the `--extra-index-url` line and the `torch==2.11.0+cu128` line,
> save it as `requirements-nocuda.txt`, and install that.

### What you just installed

`requirements.txt` was audited against every `import` in `src/` and `tools/` on
2026-09-05, and the fourteen packages nothing imported were removed — the file's
closing comment block lists each one and why. Two entries still deserve a note,
because they look wrong in opposite directions:

- **`matplotlib`** — looks unused; isn't. No file imports it, but scikit-learn's
  `ConfusionMatrixDisplay` (used at the end of `intent/train.py`) *draws* through
  matplotlib while only declaring it an optional extra. Drop it and training
  crashes on `savefig()` after the fine-tune has already run.
- **`evdev`** — looks used; barely is. `utils/hotkey.py` imports it, but that file
  has been dormant since JANET became always-listening and nothing imports *it*.
  It's pinned so the file still loads. Delete both together if you want it gone.

- **`opencv-python`** — the one forward-looking pin, kept on purpose: vision is
  being tested right now, but that work is **gitignored**, so it isn't in your
  clone and nothing you can see imports `cv2`. Install it anyway — it's what the
  next feature is built on.

---

## 5. The intent classifier (downloads itself)

**You do not need to train anything.** The model's config and tokenizer are
committed, and its 256 MB `model.safetensors` is a
[release asset](https://github.com/Samesh-Deshmukh/JANET/releases/tag/intent-model-v1)
that JANET fetches automatically the first time the classifier loads. You'll see
a progress bar once, and never again.

Why not just commit the weights? They're past GitHub's 100 MB per-file limit.
Git LFS can hold them, but its free tier allows about four clones a month before
everyone's `git clone` starts failing — release assets have no bandwidth cap on
a public repo.

Trigger the download now rather than mid-conversation:

```bash
cd src
python -c "from intent.classifier import predict; print(predict('what time is it'))"
```

`('TIME', 0.91...)` means you're done — skip to
[§6](#6-run-a-local-llm--janets-voice).

Useful switches:

| Variable | Effect |
|---|---|
| `JANET_NO_DOWNLOAD=1` | never fetch; fail with "train it instead". For strictly offline installs. |
| `JANET_INTENT_MODEL_URL=…` | fetch from your own mirror or fork instead. |

The download is verified against a SHA-256 and written atomically, so an
interrupted transfer can't leave a half-file that fails mysteriously later. And
it only ever runs when `model.safetensors` is **missing** — if you train your
own, it is never overwritten.

### Retraining it yourself (optional)

The full labelled dataset is committed too, so you can rebuild the model from
scratch — worth doing if you [add your own phrasings](#optional-teach-it-your-own-phrasings):

```bash
cd src
python -m intent.train
```

This downloads `distilbert-base-uncased` (~268 MB, cached in
`~/.cache/huggingface`), fine-tunes it on the labelled dataset in
`data/text/{train,val}/`, prints per-class metrics and a confusion matrix, and
**overwrites** `data/models/intent-distilbert/`.

**What to expect** — measured on this repo, on an RTX 5060 Ti:

| | |
|---|---|
| Training data | 2,005 examples across 13 labels |
| Validation data | 504 examples |
| Epochs | 4 |
| Time | **16 seconds** on a GPU. Budget 10–20 minutes on a CPU. |
| Validation accuracy | **97.2%** (macro F1 0.974) |

Anything in that ballpark means it worked. The run ends with:

```
accuracy: 0.972
macro F1: 0.974
...
confusion matrix saved -> .../data/models/intent-distilbert/confusion_matrix.png
model saved -> .../data/models/intent-distilbert
```

> **A warning you can ignore.** Recent `transformers` versions print
> `There were missing keys in the checkpoint model loaded: [...LayerNorm.weight...]`
> and a matching "unexpected keys" line about `LayerNorm.gamma/beta`. That's a
> parameter-renaming notice, not a failure — the final accuracy printed
> underneath is the real answer.

### Optional: check the dataset first

```bash
python data/text/validate.py     # from the repo root
```

Enforces the format contract: one utterance per line, the filename is the label,
`#` comments and blank lines ignored.

### Optional: reclaim 767 MB after retraining

Only relevant if you ran the training command above — a fresh clone never has
these. Training leaves per-epoch checkpoints behind, and the saved model doesn't
need them: 256 MB of that is a byte-identical copy of `model.safetensors`, and
the other 511 MB is optimizer state whose only use is resuming an interrupted
run. They're gitignored for exactly this reason.

```bash
rm -rf data/models/intent-distilbert/_checkpoints
```

That takes the directory from ~1 GB back down to ~257 MB.

### Optional: teach it your own phrasings

The dataset is plain text — one utterance per line, one file per label under
`data/text/train/` and `data/text/val/`. Add the way *you* actually talk, re-run
the two commands above, and JANET picks up the new model on next start. This is
the cheapest way to make it understand you better.

---

## 6. Run a local LLM — JANET's voice

JANET's handlers don't speak. They return **facts**, and a local language model
turns those facts into the sentence you actually hear. So without a model, JANET
still runs and still works — it just falls back to flat canned strings and tells
you so, once.

JANET talks over the **OpenAI-compatible `/v1` HTTP API**, so the backend is
nothing but a URL. Two options:

### Option A — Ollama (easiest, works everywhere)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b        # or qwen3:4b / llama3.2:3b on a smaller machine
ollama serve
```

Then put this in your `.env` (see [§7](#7-configure-env-optional)):

```bash
JANET_LLM_URL=http://127.0.0.1:11434/v1
JANET_LLM_MODEL=qwen3:14b
```

### Option B — llama.cpp (what the author runs; faster)

Build or install [llama.cpp](https://github.com/ggml-org/llama.cpp) (on macOS,
`brew install llama.cpp`), download a GGUF from Hugging Face, and serve it:

```bash
llama-server -m /path/to/qwen3-14b-instruct-q4_k_m.gguf -ngl 999 -c 8192 --port 8081
```

`-ngl 999` means "put every layer on the GPU" and is the single most important
flag here — it works for **Metal on Apple Silicon** exactly as it does for CUDA.
This is JANET's **default** backend — `http://127.0.0.1:8081/v1` — so no `.env`
entry is needed if you use it.

> **Getting your GPU back.** A server holding a 14B model sits on ~10 GB until
> you stop it, which is a problem the day you want to play something.
> `tools/janet-gpu.sh status|start|stop|restart` handles that, and it stops the
> **systemd user unit** rather than the process — if you run llama-server under
> one with `Restart=always`, killing the PID just brings it back in five seconds
> looking exactly like VRAM that never freed. Set `JANET_LLM_UNIT` if your unit
> isn't named `llama-qwen.service`; the script also flags the classic
> two-servers-on-one-model case that quietly halves your speed.

### Which model for your hardware

| VRAM (or unified memory) | Suggested model | Reply latency |
|---|---|---|
| 16 GB | Qwen3 14B, Q4_K_M (~9.7 GB) | 1–3 s |
| 8–12 GB | Qwen3 8B, Q4_K_M | 1–3 s |
| 6 GB | Qwen3 4B or Llama 3.2 3B | 1–3 s |
| CPU only | Llama 3.2 3B, Q4_K_M | 10–40 s — slow, but it works |

On Apple Silicon, read the left column as **unified memory** and leave headroom
for the rest of the system — a 16 GB M-series comfortably runs an 8B, and a 14B
is tight rather than impossible.

> **Do not run llama.cpp and Ollama at the same time on one GPU.** The second
> one loads a *second copy* of the weights into whatever VRAM is left, gets
> pushed onto the CPU, and every reply becomes 3–5× slower. Measured on the
> author's machine: 1.1–3.0 s became 6–7 s. If JANET ever feels inexplicably
> sluggish, check `nvidia-smi` for two processes holding the same model before
> you check anything else.

### Check it's up

```bash
curl -s http://127.0.0.1:8081/v1/models      # llama.cpp
curl -s http://127.0.0.1:11434/v1/models     # Ollama
```

---

## 7. Configure `.env` (optional)

```bash
cp .env.example .env
```

**Nothing in here is required.** Listening, transcription, the intent gates,
time, date, timers, alarms, reminders, maths, volume control and the LLM all
work with an empty `.env`. These keys only switch on the integrations that reach
outside your machine — calendar, weather, smart home, email.

Every key is documented inline in [`.env.example`](.env.example). The ones worth
knowing on day one:

| Key | What it does |
|---|---|
| `JANET_LLM_URL` | Which model server to talk to. Only needed for Ollama. |
| `JANET_WHISPER_MODEL` | `small` (default), or `tiny` if you're on a CPU and startup drags. |
| `JANET_PIPER_VOICE` | Which voice JANET speaks in — `en_US-amy-medium` by default. List them all with `python -m piper.download_voices` (no arguments). |
| `JANET_PIPER_SPEED` | Speaking pace, as a duration multiplier — **lower is faster**. Unset means the voice's own trained pace. Try `0.85`. |
| `JANET_TTS=espeak` | Fall back to the old robotic espeak-ng voice and skip Piper entirely. |
| `JANET_DEBUG_AUDIO=1` | Saves every captured utterance as a WAV. The fastest way to debug mishearings — see [§11](#11-troubleshooting). |
| `JANET_CALENDAR=demo` etc. | Fake backends with sample data, no accounts needed. |

### Try the integrations with no accounts at all

Every outside-world integration ships a fake "demo" backend. This is how the
project is tested, and it's the best way to see what JANET can do before wiring
up real credentials:

```bash
cd src
JANET_CALENDAR=demo JANET_WEATHER=demo JANET_SMART_HOME=demo JANET_EMAIL=demo python main.py
```

Now "what's on my calendar today", "is the kitchen light on", and "do I have any
new email" all answer with believable made-up data.

---

## 8. First run

Two things matter about *how* you start it:

- **Run as your normal user, never `sudo`.** *(Linux/WSL)* A per-user PipeWire
  microphone is unreachable as root, and capture comes out silent — with no error
  message. On macOS the equivalent trap is Terminal's microphone permission; see
  [§2](#macos).
- **Run from inside `src/`.** Modules import each other by top-level package
  (`from audio.audio import frames`), so `src/` must be the working directory.
  This is the one that catches everyone.

```bash
source venv/bin/activate        # Windows (native): venv\Scripts\activate
cd src
python main.py
```

**The first run downloads Whisper** (`small` is 484 MB, into `~/.cache/whisper`)
**and the Piper voice** (~63 MB, into `~/.local/share/piper-voices`), and will
sit there for a minute. Every run after that is offline. Silero VAD needs no
download — it ships inside the pip package.

You should see:

```
⏳ Warming up models...
⏳ Loading Silero VAD model (first time only)...
✅ Silero VAD ready.
⏳ Loading Whisper 'small' model (first time only)...
✅ Whisper model ready.
⏳ Loading Piper voice 'en_US-amy-medium' (first time only)...
✅ Piper voice ready.
✅ Ready.
JANET is running (always-listening). Press Ctrl-C to quit.
🎤 Listening continuously at 16000 Hz. (Ctrl-C to quit.)
```

Then just talk — there's no wake word and no key to hold. Say:

> **"Janet, what time is it?"**

and you should get:

```
🗣  You said: Janet, what time is it?
🧠 Intent: TIME (91%)
🛡  Score: 115 (janet +50, question +40, keyword +25) → addressed
⚙️  Reply: It's 8:33 PM.
💭 Why: The time is given in the facts.
```

`Ctrl-C` quits.

### Reading that output

Every utterance prints the decision trail, which is most of what you need to
debug JANET:

| Line | Meaning |
|---|---|
| `🗣  You said:` | What Whisper heard. If this is wrong, nothing downstream can save it. |
| `🧠 Intent:` | The classifier's label and confidence. `NONE`, or below 50% → silent. |
| `🛡  Score:` | The addressing gate, with the signals that earned it. Below 40 → silent. |
| `⚙️  Reply:` | What it actually said. |
| `💭 Why:` | JANET's own one-line reasoning for that reply. |
| `🛡  … → ignored` | A gate said no, and the utterance wasn't worth a model call. Silence. |
| `🤔 … → not for me` | A gate said no, the LLM was asked to double-check, and agreed. Silence. |
| `🤔 … → rescued` | The gates said no but the LLM recognised a follow-up and overruled them. |

**Silence is a feature.** JANET listens continuously, so it deliberately ignores
anything that doesn't look like a request aimed at it. Saying its name is the
strongest signal you can give it (+50 on the score).

---

## 9. Verify the install

Two check suites, both using demo backends, so neither needs credentials or a
network:

```bash
# from the repo root
export JANET_CALENDAR=demo JANET_WEATHER=demo JANET_SMART_HOME=demo JANET_EMAIL=demo
./venv/bin/python tools/smoke.py        # 30 checks, ~1 min — the fast gate
./venv/bin/python tools/full_check.py   # 69 checks — every intent, gate and guard
```

Exit code 0 means everything passed; a non-zero exit is the number of failures.

These run the **real** pipeline, not mocks, so they need the trained classifier
from step 5. Some checks also exercise the LLM — if no model server is running,
expect the fallback-path checks to be the ones that complain.

---

## 10. What your clone does *not* contain

Several directories are gitignored — some because they're private, some because
they're machine-local, some because they're large. If a document references
something you can't find, this is why.

| Path | What it was | What you do |
|---|---|---|
| `data/models/intent-distilbert/model.safetensors` | the 256 MB trained weights | **Nothing — JANET downloads it on first run** from a [release asset](https://github.com/Samesh-Deshmukh/JANET/releases/tag/intent-model-v1) ([§5](#5-the-intent-classifier-downloads-itself)). The directory's config and tokenizer *are* committed, so only this one file is fetched. |
| `data/models/` (anything else) | other trained models, and `intent-distilbert/_checkpoints/` | Machine-local. Checkpoints are 767 MB of resumable optimizer state, regenerated by training. |
| `~/.cache/whisper` | Whisper weights | Downloaded automatically on first run. |
| `~/.local/share/piper-voices` | the ~63 MB Piper voice — JANET's actual voice | Downloaded automatically on first run. `JANET_NO_DOWNLOAD=1` refuses; fetch it by hand with `python -m piper.download_voices en_US-amy-medium --download-dir ~/.local/share/piper-voices`. |
| `.env` | real credentials | `cp .env.example .env`. Optional. |
| `data/state/` | saved alarms, timers, reminders | Created at runtime. Yours will be empty. |
| `data/transcripts/` | a JSONL record of every conversation | Created at runtime. Private by definition. |
| `data/debug_audio/` | recordings of the author's voice | Only created with `JANET_DEBUG_AUDIO=1`. |
| `docs/` | the codebook, design notes, journey write-ups | Not distributed yet. `README.md` and this file are the public docs. |
| `Notes/` | the design spec, the build roadmap, the philosophy doc | Author's local working notes. |
| `CLAUDE.md`, `.claude/` | AI-assistant instructions for this repo | Local tooling config. |
| `tests/` | scratch space, **not** a test suite | Use `tools/smoke.py` and `tools/full_check.py` instead — those are committed. |
| `venv/` | the virtualenv | You create it in [§4](#4-create-the-virtualenv-and-install-python-packages). |

**What you *do* get** is everything needed to run and to build the rest: all of
`src/`, the intent classifier's config and tokenizer (its weights arrive on
first run), the full labelled training dataset in `data/text/`, both check
suites in `tools/`, `requirements.txt`, and `.env.example` with every setting
documented inline.

---

## 11. Troubleshooting

### JANET won't start

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: intent model not found at .../intent-distilbert` | The model's `config.json` is missing — an incomplete clone | Re-clone, or build it with `cd src && python -m intent.train` |
| `RuntimeError: could not download the intent model` | No internet on first run, or the release URL is unreachable | Connect and re-run — it resumes from scratch safely. Offline? Train it instead: `cd src && python -m intent.train` |
| `RuntimeError: the downloaded intent model failed its checksum` | The transfer was truncated or corrupted | Just re-run. The bad file is deleted automatically, never left behind. |
| `ModuleNotFoundError: No module named 'audio'` | Run from the wrong directory | `cd src` first — see [§8](#8-first-run) |
| `OSError: [Errno -9996] Invalid input device` | No microphone that PortAudio can see | Check `pactl list sources short`; unplug/replug; confirm you're **not** root |
| `ImportError: libportaudio.so.2` | PortAudio missing at the system level | Install it — [§2](#2-install-the-system-packages) |
| Killed during `pip install` | Out of RAM while unpacking the CUDA wheel | `pip install --no-cache-dir -r requirements.txt` |

### JANET starts but never hears me

Work through this in order — it's ordered by how often each one is the answer:

1. **Are you root?** *(Linux/WSL)* `whoami`. If it says `root`, that's the bug. A
   per-user PipeWire mic is invisible to root and capture is silently empty.
2. **Did you grant microphone permission?** *(macOS)* System Settings → Privacy &
   Security → Microphone → enable it for Terminal (or iTerm/VS Code, whichever
   you launch from). macOS hands the app **silence**, not an error, when this is
   off — so it looks exactly like a broken mic.
3. **Is the mic actually working?** `pactl list sources short`, then record a
   test: `parecord --channels=1 /tmp/t.wav` (Ctrl-C after speaking) and
   `paplay /tmp/t.wav`. On WSL2, an empty source list usually means
   `wsl --shutdown` and reopen.
4. **Can Python see any input device at all?** Platform-independent:
   ```bash
   python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name'], p.get_device_info_by_index(i)['maxInputChannels']) for i in range(p.get_device_count())]"
   ```
   Anything with `maxInputChannels > 0` is a usable mic. An empty list is a
   system problem, not a JANET problem.
5. **Is the right input selected?** JANET uses your *default* source. Change it
   in your desktop's sound settings, or with `pactl set-default-source <name>`.
6. **Are you loud enough?** Silero VAD needs a real speech signal. Raise the
   input volume: `pactl set-source-volume @DEFAULT_SOURCE@ 80%`.

### JANET hears me but says nothing

That's usually **correct behaviour** — a gate decided you weren't talking to it.
The line it printed tells you which one, and both name the reason inline:

| What you see | Which gate | What to do |
|---|---|---|
| `🛡  Score: 25 (keyword +25) → ignored` | the addressing scorer | Say **"Janet"** at the start — that's +50 on its own, and it's the single most reliable fix. Phrasing it as a question or a command adds +40. |
| `🛡  Intent: NONE (72%) → ignored` | the classifier | It read your sentence as not-a-request. Rephrase it as a direct instruction. |
| `🛡  Intent: TIMER (43%) → ignored` | the classifier's confidence floor | It's under 50% sure. Use plainer phrasing — polite indirect requests ("could you possibly…") score badly. |
| `🤔 … → not for me (…)` | the LLM's second opinion | It read the whole conversation and still said no. The reason it gives is its own words. |

If JANET's name is what's not landing, check the `🗣  You said:` line first —
mishearing "Janet" is the most common root cause of all of these.

### JANET mishears me constantly

Turn on audio debugging — this separates two completely different problems that
look identical from the transcript:

```bash
cd src && JANET_DEBUG_AUDIO=1 python main.py
```

Every utterance is saved to `data/debug_audio/` with its duration printed.

- A full sentence logged as **`0.8s`** → a **capture** problem. The mic or VAD
  only caught a fragment. Fix the input level or your distance from the mic.
- **`3.0s`** that still transcribes as nonsense → a **model** problem. Try a
  bigger Whisper: `JANET_WHISPER_MODEL=medium`.

The fixes live in different files, so don't guess — look at the duration first.

If JANET's *name* is what keeps getting mangled ("In January", "planet",
"Janet Woods"), that's the known worst case: losing the name costs +50 on the
score, so a perfectly good command scores 0 and gets dropped. `audio/stt.py`
already primes Whisper's decoder with JANET's vocabulary to fight this; adding
your own common phrasings to `INITIAL_PROMPT` there helps further.

### Every reply is slow

| Symptom | Likely cause |
|---|---|
| Long pause before *every* reply | The LLM. Run `tools/janet-gpu.sh status` — it prints what's holding the card and flags the usual culprit: two servers loaded with the same model, which pushes one of them onto the CPU. |
| Slow only on the first utterance | Normal — models load at startup, but a cold page cache still costs the first run. |
| Slow *speaking*, not thinking | Playback is ~8.5 s for a typical answer — the biggest single cost in the pipeline, and not something a faster synthesiser fixes. Piper *renders* a sentence in ~0.1 s; the rest is the speaking itself. |
| Deliberate ~11–20 s pauses | You enabled `JANET_DEEP_THINKING=1`. |

### Replies sound flat and canned

JANET says so, once: your LLM server isn't reachable. Check
`JANET_LLM_URL` against what's actually running — [§6](#6-run-a-local-llm--janets-voice).

### JANET sounds robotic

That's espeak-ng, which means Piper didn't load and JANET fell back. It prints
the reason **once**, at the moment it first happens:

```
⚠  Piper unavailable (...) — falling back to espeak-ng.
⚠  JANET will sound robotic until this is fixed.
```

Scroll back for that line — it names the cause. The usual ones:

| What the warning says | Fix |
|---|---|
| `... is missing from ~/.local/share/piper-voices and JANET_NO_DOWNLOAD is set` | Unset `JANET_NO_DOWNLOAD`, or download it by hand: `python -m piper.download_voices en_US-amy-medium --download-dir ~/.local/share/piper-voices` |
| `No module named 'piper'` | `pip install -r requirements.txt` again — `piper-tts` is new. |
| A voice name that doesn't exist | Check your `JANET_PIPER_VOICE` against `python -m piper.download_voices` (no arguments). |

If you see no warning at all, check whether `JANET_TTS=espeak` is set in your
`.env` — that selects espeak deliberately and says nothing.

### JANET answers the television

Both gates are tuned against this, but nothing is perfect. Move the mic away
from the speakers, or lower the input gain so background audio doesn't reach
speech level. Utterances longer than 40 words skip the LLM rescue check
specifically because nobody speaks a 40-word command.

### JANET talks to itself

It shouldn't — the capture thread discards frames while JANET is speaking. If
you enabled `JANET_BARGE_IN=1` **without** echo cancellation, that's the cause.
Either unset it, or load the canceller:

```bash
pactl load-module module-echo-cancel     # not persistent across reboots
```

---

## 12. Disk usage and uninstalling

Where the ~10 GB goes:

| | Size | Removable? |
|---|---|---|
| `venv/` | ~7.9 GB | yes — delete and recreate |
| `~/.cache/whisper/small.pt` | 484 MB | yes — re-downloads on next run |
| `~/.local/share/piper-voices/` | 63 MB per voice | yes — re-downloads on next run |
| `~/.cache/huggingface` | ~268 MB | yes — re-downloads on next train |
| `data/models/intent-distilbert/` | 268 MB | only by retraining |
| `data/models/intent-distilbert/_checkpoints/` | 767 MB | **yes, safe to delete now** |
| Your GGUF / Ollama model | 2–10 GB | wherever you put it |

To uninstall completely:

```bash
rm -rf JANET/ ~/.cache/whisper ~/.cache/huggingface ~/.local/share/piper-voices
# and, if you installed it: ollama rm qwen3:14b
```

JANET writes nothing outside its own directory and those three caches — except
`~/janet-workspace`, which only exists if you asked it to create files, and
`~/.config/janet/` if you set up Google Calendar.

---

## 13. macOS, Windows, and everything else

Being straight about it: **JANET is developed on Linux and only tested there.**
But it is much closer to portable than that sounds, because almost nothing in it
is platform-specific.

### What's already portable

Microphone capture (PyAudio), Silero VAD, Whisper, the DistilBERT classifier,
every parser, every handler, the confirmation gate, persistence, the LLM client
— all of it is plain Python that runs anywhere. That's ~70 of the 74 files.

### What isn't

**Two files**, and both are short:

| File | What it uses today | Public interface to keep |
|---|---|---|
| `audio/tts.py` | Piper (portable pip package) → raw PCM → `paplay`; `espeak-ng` → temp WAV → `paplay` as the fallback | `say(phrase)` — one function, blocking |
| `utils/volume.py` | `pactl` for get/set/mute | `get_volume()`, `set_volume(pct)`, `change_volume(delta)`, `is_muted()`, `set_muted(bool)` |

Nothing else in the codebase knows how either one works. `audio/speaker.py` is
the only caller of `say()`, and `actions/system_action.py` the only caller of
`volume.py`. Match those signatures and the rest of JANET can't tell.

One rule to preserve when you port `volume.py`: **every function returns `None`
on failure rather than raising.** The handler speaks "I couldn't change the
volume" off the back of that. A port that throws will crash a conversation.

There's a third, softer gap: `ai_core/sandbox.py` uses **bubblewrap**, which is
Linux-only. On macOS and Windows JANET simply declines to run code — it raises
`SandboxUnavailable` rather than falling back to running things unsandboxed,
which is the correct behaviour, so nothing is broken. You just don't get the
agent features from [JANET as an agent](README.md#janet-as-an-agent).

---

### Windows 11 — use WSL2, honestly

The best Windows path is not to port anything. **WSLg** (bundled with current
WSL2) provides a PulseAudio server with both output and microphone passthrough,
so `espeak-ng`, `paplay` and `pactl` all work exactly as on native Linux —
zero code changes. CUDA passes through too, with a current Windows-side driver.

Setup is in [§2](#2-install-the-system-packages). This is genuinely the
lowest-effort route, and it's what I'd recommend before writing a line of
porting code.

### Windows, natively

If you'd rather not use WSL, the two files above are the whole job.

**TTS** — this got easier when JANET moved to Piper. `piper-tts` is a pip
package with Windows wheels, so **the synthesiser ports itself** and you keep
JANET's actual voice; only the *player* needs replacing, and `winsound` is in
the standard library:

```python
# audio/tts.py — Windows
import os, tempfile, wave, winsound
from piper import PiperVoice

_voice = PiperVoice.load(r"C:\piper-voices\en_US-amy-medium.onnx")

def say(phrase):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        with wave.open(wav_path, "wb") as wf:
            _voice.synthesize_wav(phrase, wf)
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)   # blocking
    finally:
        os.unlink(wav_path)
```

You lose the streaming that the Linux version gets from piping into `paplay`,
which costs about 0.1 s before speech starts — not worth chasing. Keep `say()`
blocking, though: that part is not cosmetic (see the rule below).

**Volume** — needs [`pycaw`](https://github.com/AndreMiras/pycaw)
(`pip install pycaw comtypes`), which wraps the Core Audio API. Remember the
return-`None`-on-failure rule.

Everything else installs normally: PyAudio and PyTorch both publish Windows
wheels, so [§4a](#4a-you-have-an-nvidia-gpu) works unchanged with an NVIDIA card.

### macOS

macOS actually has *better* built-in tools for both files than Linux does.

**TTS** — the smallest change in the whole port. `piper-tts` has macOS wheels
(Intel and Apple Silicon), so keep JANET's real voice and swap only the player:
`afplay` for `paplay`. It reads a WAV from stdin, so the streaming design
survives too.

```python
# audio/tts.py — macOS
import subprocess
from piper import PiperVoice

_voice = PiperVoice.load("~/.local/share/piper-voices/en_US-amy-medium.onnx")

def say(phrase):
    player = subprocess.Popen(["afplay", "-"], stdin=subprocess.PIPE)
    for chunk in _voice.synthesize(phrase):
        player.stdin.write(chunk.audio_int16_bytes)
    player.stdin.close()
    player.wait()                       # blocking; that's the contract
```

(If you'd rather not carry a 63 MB voice model, macOS's built-in `say` is one
line — `subprocess.run(["say", phrase], check=True)` — and sounds far better
than espeak-ng. You just won't sound the same as everyone else's JANET.)

**Volume** — `osascript` covers the whole interface:

```python
# utils/volume.py — macOS, sketch
def _osa(script):
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None                      # never raise — see the rule above
    return out.stdout.strip() if out.returncode == 0 else None

def get_volume():
    out = _osa("output volume of (get volume settings)")
    return int(out) if out and out.isdigit() else None

def set_volume(percent):
    return _osa(f"set volume output volume {max(0, min(100, int(percent)))}")
```

Two more macOS notes:

- **The `+cu128` PyTorch pin won't resolve.** Use
  [§4b](#4b-you-have-no-gpu-an-amdintel-gpu-or-youre-on-macos). On Apple Silicon the default
  wheel includes **MPS**, so Whisper and the classifier get GPU acceleration;
  llama.cpp has excellent Metal support, so the LLM is genuinely fast on an M-series
  chip. macOS is a better JANET machine than the CPU-only advice elsewhere in this
  guide suggests.
- **Terminal needs microphone permission** (System Settings → Privacy & Security →
  Microphone). Without it you get silence and no error — see
  [§11](#11-troubleshooting).

### Anything else

BSD should work like Linux if you have PulseAudio and PortAudio. Beyond that
you're into "does it have Python 3.11, a sound server, and 10 GB free" territory
— and if the answer is yes, the two files above are still the only thing
standing between you and a talking assistant.

---

## Where to go next

- **[README.md](README.md)** — what JANET is, what it can do, and why it's built
  the way it is.
- **[`.env.example`](.env.example)** — every setting, documented inline.
- **`tools/smoke.py`** — the 30-check suite. Reading it is a fast way to see
  what the system is expected to do.
