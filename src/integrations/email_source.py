# src/integrations/email_source.py
"""Backend-agnostic email model. Handlers depend on this, not on IMAP/Gmail.

READ-ONLY BY DESIGN. This interface can count and list mail; there is
deliberately no send/reply/delete. Sending is outward-facing and irreversible,
and JANET has no multi-turn confirmation flow yet — a mis-transcription must
never be able to mail a real person. Sending gets added only after a
confirmation step exists, and it would be a separate module (SMTP), not this one.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Message:
    """One email, reduced to the few fields a voice assistant can speak."""
    sender: str          # friendly display name when the header has one, else the address
    subject: str
    date: datetime       # naive local time, like the rest of the app
    snippet: str = ""    # short preview of the body; may be "" (headers-only backends)
    unread: bool = True


class EmailSource(Protocol):
    def unread_count(self) -> int:
        """How many unread messages are in the inbox."""
        ...

    def recent(self, limit: int = 5) -> "list[Message]":
        """The most recent messages, newest first."""
        ...


class EmailUnavailable(Exception):
    """The mail server couldn't be reached, or it refused the login.

    An *operational* failure, not a bug: a source raises this so the handler can
    speak a friendly line without importing imaplib or knowing which backend is
    configured. Real bugs (a typo, a bad index) still propagate normally.
    """


class FakeEmailSource:
    """In-memory source (test double + demo). Holds a fixed list of Messages."""

    def __init__(self, messages):
        self._messages = list(messages)

    def unread_count(self):
        return sum(1 for m in self._messages if m.unread)

    def recent(self, limit=5):
        newest_first = sorted(self._messages, key=lambda m: m.date, reverse=True)
        return newest_first[:limit]
