# src/audio/tts.py
"""Local, offline text-to-speech — Piper, with espeak-ng as the safety net.

WHY PIPER
---------
espeak-ng is a formant synthesiser: it builds speech from rules about how vowels
and consonants sound. It is tiny, instant and completely intelligible, and it
sounds like a robot from 1985. Piper runs a small neural vocoder instead, so the
output has real prosody. Nothing else about JANET changed — it is the same
sentences, and they simply stopped sounding like a machine reading them.

The cost of that turned out to be close to nothing. Measured on this machine,
CPU only, on the sentence "It's twenty past four. You've got lunch with Alex at
one, and the kitchen lights are still on.":

    voice                       model    synthesis   audio    real-time factor
    en_US-amy-medium            63 MB      0.11s     6.08s        0.02
    en_GB-jenny_dioco-medium    63 MB      0.11s     5.21s        0.02
    en_US-lessac-high          114 MB      0.42s     5.39s        0.08

Synthesis is 12-50x faster than playback. Two things follow, and both shaped
this file:

1. **Streaming can never underrun.** We pipe raw PCM straight into `paplay`
   rather than writing a temp WAV and playing it afterwards, so sound starts on
   the first chunk instead of after the last one. Piper splits the text into
   sentences internally and yields a chunk each, so sentence two is synthesised
   while sentence one is already playing.
2. **The GPU stays out of it.** `PiperVoice.load()` accepts a CUDA provider and
   we deliberately do not use it. CPU already clears the bar by more than 10x,
   and the single biggest performance lesson on this machine is that a second
   process holding VRAM next to the 14B llama-server is what makes JANET feel
   slow (see "Which LLM backend" in CLAUDE.md). Free speed is not worth that.

Piper does its phonemisation with espeak-ng internally, but via a copy bundled
inside the wheel — so the pip install is genuinely self-contained and the
system `espeak-ng` binary is now needed only for the fallback below.

WHY THERE IS STILL A FALLBACK
-----------------------------
If the voice model is missing and cannot be downloaded, or onnxruntime will not
start, the right answer is a worse voice — not no voice. An assistant with no
screen that has gone silent cannot tell you why it went silent. So we drop back
to espeak-ng and print one loud warning.

The warning is the important half. A quiet fallback would present itself as
"Piper is installed but sounds exactly like espeak", which is a genuinely
confusing thing to debug. This follows `intent/nli_scorer.py`, which likewise
degrades to its old scorer rather than raising — and deliberately not
`ai_core/sandbox.py`, which refuses to run at all when bubblewrap is missing.
The difference is that the sandbox is a security boundary and this is not.

PLAYBACK
--------
Still `paplay`, for the same reason as before: this machine runs PipeWire, and
`paplay` talks to it natively. (pyttsx3's espeak driver shells out to `aplay`,
which is ALSA and not installed here, so playback failed silently with
"aplay: command not found".) `utils/volume.py` uses `pactl` from the same stack.

Along with `utils/volume.py`, this is one of only two files in JANET that shell
out to Linux-only tools. Keep it that way — `instructions.md` §13 promises the
macOS/Windows port is a small job, and that is only true while the
platform-specific surface stays this small. `JANET_TTS=espeak` exists partly so
a porter has a working escape hatch on day one.
"""
import os
import subprocess
import tempfile
from pathlib import Path

# Which synthesiser to use. `piper` is the real voice; `espeak` forces the old
# robotic one, which is useful when porting, when debugging audio routing, or on
# a machine where 63 MB of voice model isn't wanted. Same shape as the
# JANET_CALENDAR / JANET_WEATHER / JANET_SMART_HOME backend switches.
ENGINE = os.environ.get("JANET_TTS", "piper").strip().lower()

# Chosen by ear against five other candidates (British and American, male and
# female, medium and high quality tiers). Medium rather than high on purpose:
# high sounds marginally better and costs 4x the synthesis time and twice the
# disk, which is a bad trade for a voice you hear over a room.
VOICE = os.environ.get("JANET_PIPER_VOICE", "en_US-amy-medium")

# Outside the repo, like every other model JANET downloads. 63 MB of ONNX has no
# business in git — see intent/fetch_model.py for how that lesson was learned.
VOICE_DIR = Path(
    os.environ.get("JANET_PIPER_VOICE_DIR", "~/.local/share/piper-voices")
).expanduser()

# Speaking rate, as a duration multiplier: LOWER IS FASTER. Unset means "use the
# pace the voice was trained at", which is what amy ships and what was picked
# here. This rescales predicted phoneme durations before the vocoder runs, so it
# is not the same as speeding up a finished WAV — the pitch and timbre are
# untouched, only the pacing moves. Try 0.85 if the default feels slow.
_speed = os.environ.get("JANET_PIPER_SPEED", "").strip()
SPEED = float(_speed) if _speed else None

_voice = None            # the loaded PiperVoice, cached like stt.py's Whisper model
_warned = False          # so the fallback complains once, not once per sentence


def _get_voice():
    """Load the Piper voice once and reuse it.

    Loading costs ~0.6s, which is small but is exactly the kind of cost that
    lands on the first thing you say if it isn't paid at startup. `main.py`'s
    `_preload()` calls this for that reason, alongside Whisper and DistilBERT.
    """
    global _voice
    if _voice is None:
        from piper import PiperVoice          # imported late so JANET_TTS=espeak
                                              # never needs the package at all
        model = VOICE_DIR / f"{VOICE}.onnx"
        if not model.exists():
            _download_voice()
        print(f"⏳ Loading Piper voice '{VOICE}' (first time only)...")
        _voice = PiperVoice.load(model, download_dir=VOICE_DIR)
        print("✅ Piper voice ready.")
    return _voice


def _download_voice():
    """Fetch the voice model once, on first use.

    Same contract as intent/fetch_model.py: a one-off setup download is not a
    cloud dependency — nothing here touches the network again once the file is
    on disk — and JANET_NO_DOWNLOAD must be honoured for strictly-offline
    installs. Whisper and the NLI gate already behave this way.
    """
    if os.getenv("JANET_NO_DOWNLOAD"):
        raise RuntimeError(
            f"Piper voice '{VOICE}' is missing from {VOICE_DIR} and "
            "JANET_NO_DOWNLOAD is set, so it wasn't fetched. Download it with: "
            f"python -m piper.download_voices {VOICE} --download-dir {VOICE_DIR}"
        )
    from piper.download_voices import download_voice

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"-> Piper voice '{VOICE}' not found; downloading once (~63 MB)")
    download_voice(VOICE, VOICE_DIR)
    print(f"-> Piper voice ready at {VOICE_DIR}")


def say(phrase):
    """Speak `phrase` out loud, blocking until playback finishes.

    Blocking is load-bearing and must stay that way: `audio/speaker.py` decrements
    its `_pending` counter when this returns, and `is_speaking()` — the only
    thing stopping the capture thread transcribing JANET's own voice — is built
    on that counter. Return early and JANET starts answering itself.
    """
    print("Said :", phrase)

    if ENGINE == "espeak":
        _say_espeak(phrase)
        return

    try:
        _say_piper(phrase)
    except Exception as exc:
        global _warned
        if not _warned:
            _warned = True
            print(f"⚠  Piper unavailable ({exc}) — falling back to espeak-ng.")
            print("⚠  JANET will sound robotic until this is fixed.")
        _say_espeak(phrase)


def _say_piper(phrase):
    """Synthesise with Piper, streaming raw PCM straight into paplay."""
    voice = _get_voice()

    syn_config = None
    if SPEED is not None:
        from piper import SynthesisConfig
        syn_config = SynthesisConfig(length_scale=SPEED)

    # paplay is started before the first chunk exists, which is why we take the
    # rate from the voice config rather than from a chunk: it lets the audio
    # device open while the first sentence is still being synthesised.
    player = subprocess.Popen(
        [
            "paplay",
            "--raw",
            f"--rate={voice.config.sample_rate}",
            "--format=s16le",
            "--channels=1",
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for chunk in voice.synthesize(phrase, syn_config=syn_config):
            player.stdin.write(chunk.audio_int16_bytes)
        player.stdin.close()
    except BrokenPipeError:
        # paplay died mid-sentence (device unplugged, server restarted). Nothing
        # useful to write to; fall through and let the return code report it.
        pass
    except BaseException:
        # Any other failure, including Ctrl-C: don't leave a player holding the
        # audio device open with a half-finished sentence in its buffer.
        player.kill()
        player.wait()
        raise

    if player.wait() != 0:
        raise RuntimeError(f"paplay exited {player.returncode}")


def _say_espeak(phrase):
    """The original path: espeak-ng writes a temp WAV, paplay plays it.

    espeak-ng has no stdout streaming mode worth using, so this one keeps the
    temp file. It only runs as a fallback now, where latency hardly matters.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        subprocess.run(["espeak-ng", phrase, "-w", wav_path], check=True)
        subprocess.run(["paplay", wav_path], check=True)
    finally:
        os.unlink(wav_path)


if __name__ == "__main__":
    import sys

    say(" ".join(sys.argv[1:]) or "Piper is working. This is what I sound like now.")
