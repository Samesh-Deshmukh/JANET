# src/actions/email_action.py
"""EMAIL intent handler — checking mail by voice, and replying to the latest.

MOSTLY READ. The one write is "reply to the most recent message", and it is
gated twice over:
  1. Nothing is sent until utils/confirm gets a spoken "yes" — the send lives
     inside the callable handed to confirm.request and nowhere else. The
     question repeats the whole reply back, so a mis-transcription is obvious
     BEFORE it leaves the machine.
  2. The recipient comes from the message itself, never from the transcript.

Why reply-only, and no "email Bob at bob@example.com": an address dictated by
voice cannot be trusted ("at" vs "@", spelled-out domains, Whisper's habit of
inventing plausible words) and JANET has no contacts book to check it against.
The failure is silent and unrecoverable — mail to a real stranger, in your
name. Replying has no such hole: the address was supplied by the mail server.
So reply is the whole of the send feature, on purpose, not as a first step.

Privacy: answers are deliberately terse. Full bodies and email addresses are
never spoken aloud — a room can have other people in it. That applies to the
confirmation too: it names the sender ("reply to Alex Chen"), not the address.
"""
from integrations.email_factory import get_email_source
from integrations.email_source import EmailUnavailable
from utils import confirm

NOT_CONFIGURED = "Email isn't set up yet."
SEND_NOT_CONFIGURED = "Sending isn't set up yet."
UNREACHABLE = "I couldn't reach your email."
SEND_FAILED = "I couldn't send that reply."
EMPTY = "Your inbox is empty."
NOTHING_TO_REPLY_TO = "There's no email to reply to."
NO_REPLY_ADDRESS = "I don't have a reply address for that message."
ASK_FOR_BODY = "What should the reply say?"

# How many messages we'll name in one spoken summary. Three is about the limit
# of what someone can follow by ear before it becomes noise.
SPOKEN_LIMIT = 3

# Only read a preview aloud if it's this short — anything longer is a body, and
# bodies don't belong in a spoken answer.
SNIPPET_LIMIT = 100

# Phrases that mean "just the one newest message". Checked before the count and
# summary branches because they're the most specific: "read my last email" also
# contains "email". (The reply branch below is checked before all of these.)
_LATEST_WORDS = ("last email", "latest email", "last mail", "latest mail",
                 "most recent email", "most recent mail", "newest email")

# Phrases that mean "just give me the number".
_COUNT_WORDS = ("how many", "unread", "any new", "new email", "new mail",
                "check my email", "check my mail", "check my inbox",
                "check email", "check mail")

# Words that mean "this is a reply, not a question". Checked before everything
# else: "reply to my last email saying yes" contains "last email" too, and the
# read branch would happily win and swallow the request.
_REPLY_WORDS = ("reply", "respond", "write back")

# Where the reply body starts. Ordered most-specific first and the first one
# present wins, which is what makes "reply to that saying I'll be there" split
# at "saying" and not at "that" (splitting at "that" would send "saying I'll
# be there"). Each has surrounding spaces so "essay" can't match " say ".
_BODY_MARKERS = (" saying ", " to say ", " telling them ", " telling him ",
                 " telling her ", " that says ", " say ", " with ", " that ")


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


def _reply_body(query):
    """The text after a body marker — what the reply should actually say.

    Reads the RAW transcript, not the lowered/normalized one: this string is
    written down and read by a human, so "I'll be there at 5" must keep its
    capitals, apostrophe and digits. Only the *search* is case-insensitive —
    we find the marker in the lowered copy and then slice the original at that
    index, so both properties hold at once. Returns "" when no marker is there.
    """
    lowered = query.lower()
    for marker in _BODY_MARKERS:
        found = lowered.find(marker)
        if found != -1:
            return query[found + len(marker):].strip()
    return ""


def _spoken_body(body):
    """The body as it goes into the question, minus punctuation that would be
    read out ("...at five.?"). Only for speaking — the sent body is untouched."""
    return body.rstrip(" .!?,;:")


def _handle_reply(source, ctx):
    """Build the reply, then ASK before sending it (see utils/confirm).

    Nothing here sends. The send lives inside `send()`, which only runs if the
    very next thing the user says is a yes — that is the whole safety argument
    for this feature, so keep it that way.
    """
    body = _reply_body(ctx.query)
    if not body:
        return ASK_FOR_BODY                      # "reply to that" and nothing else
    # Cheap local checks before touching the network, so a mailbox that can't
    # send says so immediately instead of after a round trip.
    if not source.can_send():
        return SEND_NOT_CONFIGURED
    try:
        newest = source.recent(1)
    except EmailUnavailable:
        return UNREACHABLE
    if not newest:
        return NOTHING_TO_REPLY_TO
    original = newest[0]
    if not original.reply_to:
        # Nothing to address it to, and we will NOT ask the user to dictate one.
        return NO_REPLY_ADDRESS

    def send():
        """Runs only after a confirmed yes. Returns what JANET then says."""
        try:
            source.send_reply(original, body)
        except EmailUnavailable:
            return SEND_FAILED
        return f"Replied to {original.sender}."

    confirm.request(send)
    # State the reply back IN FULL. Half the value of the gate is that a
    # Whisper mishearing is audible before it becomes a real email.
    return f"Shall I reply to {original.sender} saying: {_spoken_body(body)}?"


def handle(slots, ctx):
    """EMAIL intent entry: reply, unread count, inbox summary, or latest message.

    `slots` is unused — EMAIL carries none. Like the CALC and GENERAL handlers,
    the phrasing is read from the raw transcript on ctx.query.
    """
    source = get_email_source()
    if source is None:
        return NOT_CONFIGURED                    # no JANET_EMAIL / IMAP config
    query = ctx.query.lower()
    # The write branch goes first: it is the most specific request, and a reply
    # phrased as "reply to my last email saying yes" also matches the read
    # phrases below.
    if any(word in query for word in _REPLY_WORDS):
        return _handle_reply(source, ctx)
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
