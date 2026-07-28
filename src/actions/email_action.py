# src/actions/email_action.py
"""EMAIL intent handler — checking and summarising mail by voice.

READ-ONLY. JANET can tell you how much mail is waiting and who it's from; it
cannot send, reply to, or delete anything. Sending is outward-facing and
irreversible, and there is no multi-turn confirmation flow yet — one Whisper
mishearing must never be able to mail a real person. Sending comes back on the
table once a confirmation step exists.

Privacy: answers are deliberately terse. Full bodies and email addresses are
never spoken aloud — a room can have other people in it.
"""
from integrations.email_factory import get_email_source
from integrations.email_source import EmailUnavailable

NOT_CONFIGURED = "Email isn't set up yet."
UNREACHABLE = "I couldn't reach your email."
EMPTY = "Your inbox is empty."

# How many messages we'll name in one spoken summary. Three is about the limit
# of what someone can follow by ear before it becomes noise.
SPOKEN_LIMIT = 3

# Only read a preview aloud if it's this short — anything longer is a body, and
# bodies don't belong in a spoken answer.
SNIPPET_LIMIT = 100

# Phrases that mean "just the one newest message". Checked first because they're
# the most specific: "read my last email" also contains "email".
_LATEST_WORDS = ("last email", "latest email", "last mail", "latest mail",
                 "most recent email", "most recent mail", "newest email")

# Phrases that mean "just give me the number".
_COUNT_WORDS = ("how many", "unread", "any new", "new email", "new mail",
                "check my email", "check my mail", "check my inbox",
                "check email", "check mail")


def _short_subject(subject, limit=60):
    """Trim a subject line to something speakable."""
    text = subject.strip()
    while text.lower().startswith(("re:", "fw:", "fwd:")):
        text = text.split(":", 1)[1].strip()     # drop reply/forward prefixes
    # Trailing punctuation would be spoken mid-sentence ("about Dinner?, and...").
    text = text.rstrip(" .!?,;:")
    if len(text) > limit:
        cut = text[:limit]
        # back off to the last whole word so we don't end mid-syllable
        text = cut[:cut.rfind(" ")] if " " in cut else cut
    return text or "no subject"


def _count_reply(count):
    """The spoken form of an unread count, with correct singular/plural."""
    if count == 0:
        return "You have no new email."
    if count == 1:
        return "You have one unread email."
    return f"You have {count} unread emails."


def _summary_reply(messages):
    """A short 'who wants what' rundown of the newest few messages."""
    if not messages:
        return EMPTY
    parts = [f"{m.sender} about {_short_subject(m.subject)}"
             for m in messages[:SPOKEN_LIMIT]]
    if len(parts) == 1:
        return f"Your most recent: {parts[0]}."
    # "A about X, B about Y, and C about Z" — the last one gets the "and"
    return "Your most recent: " + ", ".join(parts[:-1]) + f", and {parts[-1]}."


def _latest_reply(message):
    """Sender plus subject for the newest message, and a preview if it's short."""
    line = f"Your latest email is from {message.sender}, about {_short_subject(message.subject)}."
    snippet = message.snippet.strip()
    if snippet and len(snippet) <= SNIPPET_LIMIT:
        line += f" It says: {snippet}"
    return line


def handle(slots, ctx):
    """EMAIL intent entry: unread count, inbox summary, or the latest message.

    `slots` is unused — EMAIL carries none. Like the CALC and GENERAL handlers,
    the phrasing is read from the raw transcript on ctx.query.
    """
    source = get_email_source()
    if source is None:
        return NOT_CONFIGURED                    # no JANET_EMAIL / IMAP config
    query = ctx.query.lower()
    try:
        if any(phrase in query for phrase in _LATEST_WORDS):
            newest = source.recent(1)
            return _latest_reply(newest[0]) if newest else EMPTY
        if any(phrase in query for phrase in _COUNT_WORDS):
            return _count_reply(source.unread_count())
        # Default: "who emailed me", "what's in my inbox", "read my email".
        return _summary_reply(source.recent(SPOKEN_LIMIT))
    except EmailUnavailable:
        # server down / wrong password — operational, not a bug
        return UNREACHABLE
