# src/ai_core/host.py
"""A short list of harmless commands JANET may run on the real machine.

Everything JANET executes normally goes in the [sandbox], which has no home, no
credentials and no network. But sometimes the question is *about* the real
machine — "is the model still on the GPU?", "what's uncommitted?" — and a sandbox
that can't see anything can't answer that.

So: a fixed allowlist, and three rules that matter more than the list itself.

**1. No shell.** The command is parsed with `shlex` and executed as an argv list,
never through `/bin/sh`. `ls; rm -rf ~` doesn't become two commands — it becomes
`ls` with the literal arguments `;` and `rm`, which fails harmlessly.

**2. Read-only isn't enough on its own.** `cat` is a read-only program, and
`cat ~/.ssh/id_rsa` still hands a private key to a language model that will read
it out loud. So arguments may not be absolute paths, may not contain `..`, and
may not name a secret. Commands run with the repo as their working directory.

**3. Some "safe" programs execute things.** `find -exec rm {} \\;` deletes files,
`git push` writes to GitHub. Those need per-command argument rules, not just a
program name on a list.

Anything not on the list isn't refused outright — it's offered to the sandbox
instead, where it can run without any of these worries.
"""
import os
import shlex
import subprocess
from pathlib import Path

TIMEOUT_S = 15
MAX_OUTPUT = 4000

# src/ai_core/host.py -> src/ai_core -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Programs that only ever report. A value of None means "no extra argument
# rules"; a set means "only these subcommands".
ALLOWED = {
    "ls": None, "cat": None, "head": None, "tail": None, "wc": None,
    "grep": None, "rg": None, "file": None, "stat": None, "tree": None,
    "df": None, "free": None, "uptime": None, "date": None, "pwd": None,
    "whoami": None, "hostname": None, "uname": None,
    "nvidia-smi": None, "lscpu": None, "ps": None,
    "find": None,                          # plus the flag rules below
    "git": {"status", "log", "diff", "show", "branch", "remote", "describe",
            "blame", "shortlog", "ls-files"},
}

# `find` is a program that can run other programs. Without this it is the widest
# hole on the list.
FIND_FORBIDDEN = {"-exec", "-execdir", "-ok", "-okdir", "-delete",
                  "-fprintf", "-fprint", "-fls"}

# Files inside the repo that must never be read out. `.env` holds the real IMAP,
# CalDAV and Home Assistant credentials.
SECRET_HINTS = (".env", "id_rsa", "id_ed25519", "credentials", "token",
                ".git/config", "google_token", "google_credentials")


class NotAllowed(ValueError):
    """The command isn't on the allowlist, or its arguments break a rule."""


# We never use shell=True, so these are already inert — `ls && curl evil.com`
# just hands `ls` three junk arguments. They are refused anyway: relying on "we
# don't use a shell" is subtle, and one careless refactor away from becoming a
# command-injection bug. Reject the shape, not just the mechanism.
SHELL_METACHARACTERS = set(";&|><`$\n()")


def _check_argument(arg):
    lowered = arg.lower()
    if any(hint in lowered for hint in SECRET_HINTS):
        raise NotAllowed(f"{arg!r} looks like a secret")
    if arg.startswith("-"):
        return                              # a flag, not a path
    if arg.startswith("/") or arg.startswith("~"):
        raise NotAllowed("absolute paths aren't allowed — stay in the repo")
    if ".." in Path(arg).parts:
        raise NotAllowed("'..' isn't allowed — stay in the repo")


def check(command):
    """Parse and validate. Returns the argv list, or raises NotAllowed."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:              # unbalanced quotes
        raise NotAllowed(f"couldn't parse that command: {exc}") from exc
    if not argv:
        raise NotAllowed("empty command")
    if SHELL_METACHARACTERS & set(command):
        raise NotAllowed("shell operators aren't allowed — one command at a time")

    program = os.path.basename(argv[0])
    if program not in ALLOWED:
        raise NotAllowed(f"{program!r} isn't on the read-only allowlist")

    subcommands = ALLOWED[program]
    if subcommands is not None:
        if len(argv) < 2 or argv[1] not in subcommands:
            allowed = ", ".join(sorted(subcommands))
            raise NotAllowed(f"only these {program} subcommands: {allowed}")
    if program == "find":
        for arg in argv[1:]:
            if arg.lower() in FIND_FORBIDDEN:
                raise NotAllowed(f"find {arg} can modify or execute — refused")

    for arg in argv[1:]:
        _check_argument(arg)
    return argv


def run(command):
    """Run an allowlisted command in the repo. Returns its output as text."""
    argv = check(command)                   # raises NotAllowed
    try:
        proc = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=TIMEOUT_S,
        )
    except FileNotFoundError:
        return f"{argv[0]}: not installed"
    except subprocess.TimeoutExpired:
        return f"timed out after {TIMEOUT_S}s"
    output = (proc.stdout + proc.stderr).strip()
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + "\n… (truncated)"
    return output or "(no output)"
