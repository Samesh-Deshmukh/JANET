# src/ai_core/selfmod.py
"""JANET changing its own source code — the careful path.

The flow, and every step of it exists for a reason:

    1. ASK      one-line summary, spoken. You say yes or no.
    2. DESIGN   it writes docs/janet_changes/<date>-<slug>.md and stops.
    3. REVIEW   you read it. Say "go ahead", or say what to change.
    4. CODE     it writes the change on a NEW GIT BRANCH, never on master.
    5. TEST     syntax, imports, and the full smoke suite.
    6. STOP     it reports and goes quiet. YOU read the diff, merge, restart.

The thing this design refuses to do is let a spoken "yes" put unreviewed code
into a running assistant. You can hear a summary; you cannot hear a diff. So the
spoken gate only authorises *writing a proposal*, and the code itself is
authorised by you reading it.

JANET never merges and never restarts itself. A change is inert until you act.
"""
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ai_core import llm

REPO = Path(__file__).resolve().parents[2]
CHANGES_DIR = REPO / "docs" / "janet_changes"

# A design doc should be readable in under 7 minutes. ~200 wpm, so ~1400 words.
# It is a CEILING, not a target: a one-line fix gets a one-paragraph doc.
MAX_DOC_WORDS = 1400

# A pending review keeps "conversation mode" on — every utterance gets the
# addressing check, because review chatter ("make it shorter", "no, use a dict")
# scores badly and won't land inside the normal 30s window. That is expensive, so
# it expires: without this, one open review means an LLM call per overheard
# sentence forever.
REVIEW_TIMEOUT_S = float(os.environ.get("JANET_REVIEW_TIMEOUT", "600"))   # 10 min

# Files JANET may never edit, no matter what it is asked.
#
# Without this list, "remove the confirmation gate" is a perfectly valid change
# request — and once it lands, nothing ever asks again. Every safety property in
# JANET lives in one of these files, including this one. If the owner genuinely
# wants one changed, they change it by hand; that is the point.
PROTECTED = {
    "src/utils/confirm.py",        # the spoken yes/no gate
    "src/ai_core/addressing.py",   # the "was I talked to" gate
    "src/ai_core/sandbox.py",      # the isolation
    "src/ai_core/host.py",         # the command allowlist
    "src/ai_core/workspace.py",    # the file confinement
    "src/ai_core/selfmod.py",      # this file
    ".gitignore",
}
PROTECTED_PREFIXES = (".git/", ".env")


class Refused(RuntimeError):
    """The change touches something JANET isn't allowed to change."""


def is_protected(path):
    """True if `path` (repo-relative) must never be written by JANET.

    NOTE the prefix handling. The first version used `str(path).lstrip("./")`,
    which is a trap: `lstrip` removes any leading character *in that set*, so
    ".env" became "env" and ".gitignore" became "gitignore" — and all three of
    the dotfiles on the list silently stopped being protected. Strip the "./"
    prefix explicitly instead.
    """
    normalised = str(path).replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    normalised = normalised.lstrip("/")
    return (normalised in PROTECTED
            or normalised.startswith(PROTECTED_PREFIXES))


def check_paths(paths):
    """Raise if any path is protected. Refusal is stated, never silent."""
    blocked = [p for p in paths if is_protected(p)]
    if blocked:
        raise Refused(
            f"I'm not allowed to edit {', '.join(blocked)} — those are the files "
            "that keep me asking permission. You'd have to change them yourself."
        )


@dataclass
class Change:
    """One proposed change, carried through the whole flow."""
    summary: str
    doc_path: Path | None = None
    branch: str | None = None
    files: dict = field(default_factory=dict)     # repo-relative path -> new text
    created: float = field(default_factory=time.monotonic)

    def expired(self):
        return time.monotonic() - self.created > REVIEW_TIMEOUT_S


# One at a time. Two half-reviewed changes in flight would be impossible to talk
# about out loud ("go ahead" — with which one?).
_pending: Change | None = None


def pending():
    """The change awaiting review, or None. Expired reviews are dropped here."""
    global _pending
    if _pending is not None and _pending.expired():
        print("🕑 design review expired — dropping it")
        _pending = None
    return _pending


def clear():
    global _pending
    _pending = None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "change"


def _git(*args, check=True):
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, timeout=30)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


DOC_PROMPT = (
    "You are JANET, writing a design document for a change to your OWN source "
    "code. The owner will read it and either approve it or ask for changes.\n"
    "\n"
    "Keep it under {max_words} words — it must be readable in under 7 minutes. "
    "A small change deserves a short document; do not pad it.\n"
    "\n"
    "Cover, briefly: what the change is, why it is worth doing, which files you "
    "would touch, how it could go wrong, and how it will be tested. Use plain "
    "Markdown. Be concrete about file names and function names. Do not write the "
    "code itself — this is the proposal, not the implementation.\n"
    "\n"
    "Do NOT write a top-level '# Title' heading; one is added for you. Start "
    "with the first section."
)


def write_design_doc(change, repo_context=""):
    """Ask the model for a design doc, save it, return the path."""
    CHANGES_DIR.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": DOC_PROMPT.format(max_words=MAX_DOC_WORDS)},
        {"role": "user", "content":
            f"The change: {change.summary}\n\n{repo_context}".strip()},
    ]
    thinking, doc = llm.chat(messages, max_tokens=2200, timeout=180)
    doc = _strip_fences(doc)
    words = len(doc.split())
    if words > MAX_DOC_WORDS:
        doc = " ".join(doc.split()[:MAX_DOC_WORDS]) + "\n\n… (truncated to keep it a short read)"
    path = CHANGES_DIR / f"{datetime.now():%Y-%m-%d}-{_slug(change.summary)}.md"
    path.write_text(f"# {change.summary}\n\n{doc}\n", encoding="utf-8")
    change.doc_path = path
    return path


def _strip_fences(text):
    """Models emit ```python fences even when told not to. Measured, repeatedly."""
    text = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", text, flags=re.M)
    return text.strip()


CODE_PROMPT = (
    "You are JANET, writing a change to your own source code. You will be given "
    "the approved design and the current contents of the files involved.\n"
    "\n"
    "Return the COMPLETE new contents of every file you change, in this exact "
    "format and nothing else:\n"
    "\n"
    "=== FILE: src/path/to/file.py ===\n"
    "<the entire new file>\n"
    "=== END ===\n"
    "\n"
    "Rules:\n"
    "- Return WHOLE files, never diffs or fragments. No markdown code fences.\n"
    "- Change as little as possible. Do NOT reformat, do NOT tidy, do NOT "
    "reorganise anything you were not asked to change.\n"
    "- PRESERVE every existing comment, docstring and blank line you are not "
    "explicitly changing — including the module docstring at the top of the "
    "file. Deleting documentation is a failure even if the code still works.\n"
    "- Match the surrounding style: this codebase values being understandable "
    "over clever, and its comments explain WHY, not what."
)

# The code comes back through a JSON schema rather than a text format. Two
# reasons, both measured:
#
#  * Constrained decoding SUPPRESSES the <think> block. Asked in free text, the
#    model spent 17,000 characters reasoning, hit the token ceiling, and
#    returned no code at all.
#  * The shape is guaranteed — no regex parsing a format the model may drift
#    from halfway through a long reply.
CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["path", "contents"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["files", "notes"],
}


def parse_files(payload):
    """{path: contents} from the model's structured reply."""
    return {entry["path"].strip(): _strip_fences(entry["contents"])
            for entry in payload.get("files", [])
            if entry.get("path") and entry.get("contents")}


def write_code(change, file_contents):
    """Ask the model for the new file contents. Returns {path: text}."""
    context = "\n\n".join(
        f"=== CURRENT: {path} ===\n{text}" for path, text in file_contents.items()
    )
    design = change.doc_path.read_text(encoding="utf-8") if change.doc_path else change.summary
    messages = [
        {"role": "system", "content": CODE_PROMPT},
        {"role": "user", "content": f"{design}\n\n{context}"},
    ]
    payload = llm.chat(messages, schema=CODE_SCHEMA, max_tokens=6000, timeout=600)
    files = parse_files(payload)
    check_paths(files)              # refuse protected files even here
    return files, (payload.get("notes") or "").strip()


def apply_on_branch(change):
    """Create a branch, write the files, commit. Returns the branch name."""
    check_paths(change.files)
    if not change.files:
        raise RuntimeError("no files to write")
    branch = f"janet/{datetime.now():%Y%m%d-%H%M}-{_slug(change.summary)}"
    original = _git("rev-parse", "--abbrev-ref", "HEAD")
    _git("checkout", "-b", branch)
    try:
        for relative, text in change.files.items():
            target = REPO / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        _git("add", *change.files.keys())
        _git("commit", "-m",
             f"JANET: {change.summary}\n\nWritten by JANET on request. Design: "
             f"{change.doc_path.relative_to(REPO) if change.doc_path else 'n/a'}\n"
             "Not merged — review the diff before merging.")
    except Exception:
        _git("checkout", original, check=False)
        raise
    change.branch = branch
    return branch, original


def run_tests(changed_paths):
    """Syntax, import, then the full smoke suite. Returns (ok, report).

    **What this cannot catch, and why the human review exists.** On the first
    real run, JANET's change passed every one of these — syntax clean, 30/30
    smoke — while quietly deleting the target file's entire 17-line module
    docstring and collapsing the blank lines between functions. Behaviour was
    identical, so no test could object. Tests prove a change didn't *break*
    anything; only a person reading the diff can see what it *removed*.
    """
    lines = []
    ok = True

    # 1. Syntax. The failure that matters most: a syntax error in dispatch.py
    # means JANET won't start, and you can't voice-fix something that isn't
    # running.
    for relative in changed_paths:
        if not str(relative).endswith(".py"):
            continue
        proc = subprocess.run(["python3", "-m", "py_compile", str(REPO / relative)],
                              capture_output=True, text=True, timeout=60)
        good = proc.returncode == 0
        ok &= good
        lines.append(f"syntax {relative}: {'ok' if good else proc.stderr.strip()[:200]}")

    # 2 + 3. Imports and behaviour, via the smoke suite — it imports main and
    # drives the real pipeline, so a broken import shows up here too.
    smoke = REPO / "tools" / "smoke.py"
    if smoke.exists():
        proc = subprocess.run([str(REPO / "venv" / "bin" / "python"), str(smoke)],
                              capture_output=True, text=True, timeout=600, cwd=REPO)
        tail = [l for l in proc.stdout.splitlines() if "passed" in l or "FAILED" in l]
        ok &= proc.returncode == 0
        lines.append("smoke: " + ("; ".join(tail) if tail else f"exit {proc.returncode}"))
    else:
        lines.append("smoke: tools/smoke.py missing — skipped")
    return ok, "\n".join(lines)
