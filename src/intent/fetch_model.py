"""Fetch the trained intent classifier's weights if they aren't on disk yet.

One concern: make sure `model.safetensors` exists, downloading it once if not.

WHY THIS FILE EXISTS
--------------------
The fine-tuned DistilBERT is 256 MB. That is past GitHub's 100 MB per-file
limit, so it cannot be committed. Git LFS *can* hold it, but the free tier gives
1 GB of bandwidth a month against 256 MB per clone — about four clones before
everyone's `git clone` starts failing. A **release asset** has no bandwidth cap
on a public repo, so the weights live there and this module fetches them.

Everything else in the model directory (config, tokenizer — 783 KB total) IS
committed, so a clone is only ever missing this one file.

This mirrors what JANET already does elsewhere: Whisper downloads its weights on
first use, and the NLI addressing gate pulls 1.7 GB from Hugging Face. A one-off
setup download is not a cloud dependency — nothing here phones home at runtime,
and once the file is on disk JANET never contacts the network again.

THE RULE THAT MATTERS: only download when the file is MISSING.
Someone who runs `python -m intent.train` gets their own weights, with a
different hash from the release. Any "verify the hash, re-download on mismatch"
design would silently overwrite their retrained model on every startup. So
existence is the only check; the checksum validates bytes we just fetched,
nothing else.
"""
import os
import sys
import urllib.request

# The release asset. Overridable so a fork, a mirror, or an air-gapped copy can
# point somewhere else without editing code.
DEFAULT_URL = (
    "https://github.com/Samesh-Deshmukh/JANET/releases/download/"
    "intent-model-v1/model.safetensors"
)
# Checked against what we download, to catch a truncated or corrupted transfer.
# Without this, a half-finished download surfaces as an unintelligible
# safetensors header error rather than "your file is incomplete".
EXPECTED_SHA256 = "79af5857cb63373ae695647436301a9bd94fbb4af02993ff6cbcea0799ef9677"
EXPECTED_BYTES = 267866404

WEIGHTS_NAME = "model.safetensors"


def _progress(done, total):
    """One rewriting line, so a 256 MB download doesn't look like a hang."""
    if not total:
        return
    pct = done * 100 // total
    bar = "#" * (pct // 4)
    sys.stderr.write(f"\r   [{bar:<25}] {pct:3d}%  {done // 1048576:>3d}/{total // 1048576} MB")
    sys.stderr.flush()


def ensure_weights(model_dir):
    """Make sure model_dir/model.safetensors exists. Download it once if not.

    Returns the Path to the weights. Raises RuntimeError with an actionable
    message if the file is missing and cannot be fetched — the caller turns
    that into advice about training the model instead.
    """
    import hashlib
    from pathlib import Path

    model_dir = Path(model_dir)
    target = model_dir / WEIGHTS_NAME

    # Already here — including a model the user trained themselves. Never touch it.
    if target.exists():
        return target

    if os.getenv("JANET_NO_DOWNLOAD"):
        raise RuntimeError(
            f"{WEIGHTS_NAME} is missing and JANET_NO_DOWNLOAD is set, so it "
            "wasn't fetched. Train it instead: cd src && python -m intent.train"
        )

    url = os.getenv("JANET_INTENT_MODEL_URL", DEFAULT_URL)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download beside the target, then rename. An interrupted download must
    # never leave a half-file at the real path, because the check above is
    # "does it exist" — a truncated file would be trusted forever after.
    partial = target.with_suffix(".safetensors.part")

    print(f"-> intent model weights not found; downloading once ({EXPECTED_BYTES // 1048576} MB)")
    print(f"   from {url}")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length") or EXPECTED_BYTES)
            done = 0
            with open(partial, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    _progress(done, total)
        sys.stderr.write("\n")
    except Exception as e:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the intent model from {url} ({e}). "
            "Check your connection, or train it yourself: "
            "cd src && python -m intent.train"
        ) from e

    got = digest.hexdigest()
    if got != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "the downloaded intent model failed its checksum "
            f"(expected {EXPECTED_SHA256[:16]}..., got {got[:16]}...). "
            "The download was incomplete or the file changed. Delete nothing "
            "and just re-run, or train it: cd src && python -m intent.train"
        )

    partial.replace(target)   # atomic: the file appears complete or not at all
    print(f"-> intent model ready at {target}")
    return target
