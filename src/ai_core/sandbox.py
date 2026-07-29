# src/ai_core/sandbox.py
"""Run code and commands where they cannot hurt anything.

JANET is always listening and its "permission" is a spoken yes from a pipeline
that mishears constantly. That makes arbitrary execution on the real machine a
bad trade: this account can reach `.env` credentials, ssh keys, and push rights
to GitHub. So anything JANET runs goes in here instead.

The isolation is **bubblewrap** (`bwrap`), and it is real — verified on this
machine, sandboxed code cannot see `/home` and has no DNS or network at all:

    $ bwrap ... /bin/sh -c 'ls ~'
    ls: cannot access '/home/samesh': No such file or directory
    $ bwrap ... /bin/sh -c 'getent hosts github.com'
    NO DNS/NETWORK

What the sandbox gets: a read-only system (`/usr`, `/bin`, `/lib`), a private
`/tmp`, its own PID namespace, and one writable working directory. What it does
NOT get: your home, your credentials, the network, or the ability to outlive the
call. That turns "let JANET run commands" from a footgun into something safe
enough to say yes to out loud.
"""
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

# Long enough for a real script, short enough that a runaway loop doesn't hang
# the (single-threaded) assistant for minutes.
TIMEOUT_S = float(os.environ.get("JANET_SANDBOX_TIMEOUT", "20"))

# Output is read aloud or fed back to the model. Truncate hard.
MAX_OUTPUT = 4000

# Python colours its tracebacks now, so a failing script comes back full of
# "\x1b[1;35m". That junk would land in the model's context and, worse, could be
# read aloud. Strip it at the boundary.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class SandboxUnavailable(RuntimeError):
    """bwrap isn't installed, so we have no way to run anything safely.

    Deliberately fatal to the *request*, never a fallback to running unsandboxed:
    "I can't run that safely" is the correct answer, not running it anyway.
    """


@dataclass
class Result:
    ok: bool
    exit_code: int
    output: str            # stdout and stderr, combined and truncated


def available():
    """True when we can actually sandbox. Checked before offering to run anything."""
    return shutil.which("bwrap") is not None


# The system paths a script needs to run at all. Bound READ-ONLY, and only if
# they exist — distributions differ (Arch has no /etc/alternatives, and /lib64
# is a symlink on some systems), and bwrap refuses to start if asked to bind a
# path that isn't there.
_SYSTEM_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives",
                 "/etc/ssl/certs")


def _argv(workdir, command):
    """The bwrap invocation. Every flag here is load-bearing."""
    argv = ["bwrap"]
    for path in _SYSTEM_PATHS:
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    return argv + [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # The ONE writable place, and it's a throwaway directory we made.
        "--bind", workdir, "/work",
        "--chdir", "/work",
        # No network, no IPC, no PID/user namespace sharing. This is what stops
        # a script phoning home or touching another process.
        "--unshare-all",
        # If JANET dies, nothing it started survives it.
        "--die-with-parent",
        "--new-session",          # no terminal to hijack
        "/bin/sh", "-c", command,
    ]


def run(command, files=None):
    """Run `command` inside the sandbox and return a Result.

    `files` is an optional {name: text} written into the working directory first,
    which is how code gets in — the model writes a file, then runs it.
    """
    if not available():
        raise SandboxUnavailable("bwrap is not installed")

    # A fresh directory per run: nothing carries over between calls, so one
    # script can't leave something behind for the next.
    with tempfile.TemporaryDirectory(prefix="janet-sandbox-") as workdir:
        for name, text in (files or {}).items():
            # Keep writes inside the workdir — a name like "../../x" must not escape.
            safe = os.path.basename(name)
            with open(os.path.join(workdir, safe), "w", encoding="utf-8") as handle:
                handle.write(text)
        try:
            proc = subprocess.run(
                _argv(workdir, command),
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return Result(False, -1, f"timed out after {TIMEOUT_S:.0f}s")
        output = _ANSI.sub("", proc.stdout + proc.stderr).strip()
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n… (truncated)"
        return Result(proc.returncode == 0, proc.returncode, output)


def run_python(code):
    """Write `code` to a file and run it. The common case."""
    return run("python3 main.py", files={"main.py": code})
