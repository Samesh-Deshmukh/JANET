# src/ai_core/workspace.py
"""File operations, confined to one directory.

JANET can create, read, edit and delete files — but only inside a workspace
directory, never across your whole disk. The confinement is the point: a
mis-transcribed path is a wasted file in a scratch folder rather than an
overwritten `~/.ssh/authorized_keys`.

Every path is resolved to an absolute real path and then checked to be under the
root, which is what stops `../../.ssh/id_rsa` and a symlink pointing out of the
tree. Resolution happens BEFORE the check, so a link cannot be followed out.
"""
import os
import shutil
from pathlib import Path

# Default lives outside the repo so JANET's scratch work never mixes with its
# own source — that only happens through the reviewed self-modification path.
DEFAULT_ROOT = Path.home() / "janet-workspace"

MAX_READ_CHARS = 20000        # a file's contents end up in the model's context


class OutsideWorkspace(ValueError):
    """A path resolved to somewhere outside the workspace. Always refused."""


def root():
    return Path(os.environ.get("JANET_WORKSPACE", DEFAULT_ROOT)).expanduser()


def _resolve(relative):
    """Absolute path inside the workspace, or raise.

    `strict=False` so a file that doesn't exist yet still resolves — we need to
    validate the location of a file we're about to create, not just existing ones.
    """
    text = str(relative)
    # "~/..." is contained (it becomes a literal directory called "~" inside the
    # workspace) but it clearly MEANS the home directory, and silently creating
    # a folder named "~" is worse than saying no.
    if text.startswith("~"):
        raise OutsideWorkspace("'~' means your home directory — not allowed here")
    base = root().resolve()
    base.mkdir(parents=True, exist_ok=True)
    candidate = (base / text).resolve(strict=False)
    # `is_relative_to` compares the RESOLVED paths, so symlinks and ".." have
    # already been collapsed by this point.
    if candidate != base and not candidate.is_relative_to(base):
        raise OutsideWorkspace(f"{relative!r} is outside the workspace")
    return candidate


def listing(relative="."):
    """Names in a directory, one per line."""
    target = _resolve(relative)
    if not target.exists():
        return f"{relative}: nothing there"
    if target.is_file():
        return f"{relative}: a file, {target.stat().st_size} bytes"
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return f"{relative}: " + (", ".join(names) if names else "empty")


def read(relative):
    target = _resolve(relative)
    if not target.is_file():
        return f"{relative}: no such file"
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n… (truncated)"
    return text


def write(relative, text):
    """Create or overwrite a file. Parent directories are created."""
    target = _resolve(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(text, encoding="utf-8")
    return f"{'updated' if existed else 'created'} {relative} ({len(text)} chars)"


def delete(relative):
    """Remove a file, or a directory and its contents."""
    target = _resolve(relative)
    if not target.exists():
        return f"{relative}: nothing to delete"
    # Refuse to delete the workspace itself — "delete everything" should need a
    # deliberate act outside JANET, not one misheard sentence.
    if target == root().resolve():
        raise OutsideWorkspace("refusing to delete the workspace root")
    if target.is_dir():
        shutil.rmtree(target)
        return f"deleted directory {relative}"
    target.unlink()
    return f"deleted {relative}"
