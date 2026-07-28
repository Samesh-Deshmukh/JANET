# src/integrations/email_source.py
"""Backend-agnostic email model. Handlers depend on this, not on IMAP/Gmail.

READ, PLUS ONE NARROW WRITE: replying to a message we already have.

Sending is outward-facing and irreversible, so the one write in this interface
is deliberately the *safest* one. `send_reply` takes its recipient from a
Message that the mail server itself gave us — nothing is ever addressed from a
transcript. There is still no "compose a new email to <address>": dictating an
address by voice is unreliable ("at" vs "@", spelled-out domains) and JANET has
no contacts book, so the recipient could not be checked. Replying has no such
hole, which is why it is the only send path.

Every send also goes through utils/confirm first — see actions/email_action.py.
Deleting mail is still absent, and stays absent: it is destructive and there is
nothing to undo it with.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Message:
    """One email, reduced to the few fields a voice assistant can speak.

    The last two fields are never spoken — they exist only so a reply can be
    addressed and threaded correctly. They keep their defaults so a backend that
    can't supply them (or an old fixture) still builds a valid Message.
    """
    sender: str          # friendly display name when the header has one, else the address
    subject: str
    date: datetime       # naive local time, like the rest of the app
    snippet: str = ""    # short preview of the body; may be "" (headers-only backends)
    unread: bool = True
    reply_to: str = ""   # bare address a reply goes to ("alex@example.com"), "" if unknown
    message_id: str = "" # the Message-ID header, angle brackets included, for threading


@dataclass
class SentReply:
    """A record of one reply that was sent (what FakeEmailSource collects).

    A dataclass rather than a dict so a test reads `sent[0].subject` and a typo
    raises instead of silently returning None.
    """
    to: str
    subject: str
    body: str
    in_reply_to: str = ""


def reply_subject(subject):
    """The Subject header for a reply: 'Re: ' + the original, added only once.

    Lives here, not in the SMTP module, because it is part of the reply contract
    every backend has to honour — the fake records exactly the subject the real
    sender would put on the wire, so a test of one is a test of both.
    Mail clients match threads on this prefix, so "Re: Re: Invoice" is both wrong
    and a giveaway that a bot wrote it.
    """
    text = subject.strip()
    if text.lower().startswith("re:"):
        return text                      # already a reply — leave the thread alone
    return f"Re: {text}" if text else "Re:"


class EmailSource(Protocol):
    def unread_count(self) -> int:
        """How many unread messages are in the inbox."""
        ...

    def recent(self, limit: int = 5) -> "list[Message]":
        """The most recent messages, newest first."""
        ...

    def can_send(self) -> bool:
        """True when this source is able to send a reply.

        Reading and sending are separate services (IMAP and SMTP), so a mailbox
        can be perfectly readable with no way to send. Asking up front lets the
        handler say "Sending isn't set up yet" instead of building a whole
        confirmation and only failing after the user says yes.
        """
        ...

    def send_reply(self, message: Message, body: str) -> None:
        """Send `body` as a reply to `message`. Raises EmailUnavailable on failure.

        Returns nothing: there is no useful result, and the caller already knows
        what it asked for. Must only be called after a confirmed yes.
        """
        ...


class EmailUnavailable(Exception):
    """The mail server couldn't be reached, or it refused the login/the message.

    An *operational* failure, not a bug: a source raises this so the handler can
    speak a friendly line without importing imaplib or knowing which backend is
    configured. Real bugs (a typo, a bad index) still propagate normally.
    """


class FakeEmailSource:
    """In-memory source (test double + demo). Holds a fixed list of Messages."""

    def __init__(self, messages):
        self._messages = list(messages)
        # Public on purpose: this is the whole point of the fake. A test asserts
        # on `source.sent` to see exactly what would have gone out, with no
        # network and no mail account anywhere in sight.
        self.sent = []

    def unread_count(self):
        return sum(1 for m in self._messages if m.unread)

    def recent(self, limit=5):
        newest_first = sorted(self._messages, key=lambda m: m.date, reverse=True)
        return newest_first[:limit]

    def can_send(self):
        return True                      # a fake can always "send"

    def send_reply(self, message, body):
        """Record the reply instead of sending it."""
        self.sent.append(SentReply(
            to=message.reply_to,
            subject=reply_subject(message.subject),
            body=body,
            in_reply_to=message.message_id,
        ))
