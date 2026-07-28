# src/integrations/email_factory.py
"""Choose the configured EmailSource from env: demo, IMAP, or None.

Keeping the choice here means the handler never imports a specific backend —
adding JMAP or a local Maildir later is a change to this file only. It is also
the only file that reads the JANET_* mail settings, so credentials never appear
in a backend class.

Reading and sending are configured separately (JANET_IMAP_* vs JANET_SMTP_*),
because they genuinely are separate services — a mailbox can be perfectly
readable with no way to send, and that setup should work rather than fail.
"""
import os
from datetime import datetime

from dotenv import load_dotenv

from integrations.email_source import FakeEmailSource
from integrations.demo_email import demo_messages
from integrations.imap_source import IMAPSource
from integrations.smtp_sender import SMTPSender

load_dotenv()


def _get_smtp_sender():
    """The configured SMTPSender, or None when sending isn't set up.

    Only the host is a new required setting: the login usually is the IMAP one,
    so JANET_SMTP_USER/PASSWORD fall back to the IMAP pair. That way adding
    sending to a working setup is a single line in .env.
    """
    host = os.environ.get("JANET_SMTP_HOST")
    if not host:
        return None                              # no host -> sending is off
    user = os.environ.get("JANET_SMTP_USER") or os.environ.get("JANET_IMAP_USER")
    password = (os.environ.get("JANET_SMTP_PASSWORD")
                or os.environ.get("JANET_IMAP_PASSWORD"))
    if not (user and password):
        return None                              # half-configured == not configured
    port = int(os.environ.get("JANET_SMTP_PORT", "465"))   # 465 = SMTP over SSL
    # The address mail appears to come from; usually the login, but some relays
    # allow sending as an alias.
    from_address = os.environ.get("JANET_EMAIL_FROM") or user
    return SMTPSender(host, user, password, port=port, from_address=from_address)


# Built once and reused (like calendar_factory and the Whisper/DistilBERT
# singletons). Two flags rather than one because None is a legitimate result
# ("not configured") and must be told apart from "haven't looked yet".
# This matters twice over: the demo inbox keeps its state (including what it
# "sent") in memory, and a real IMAPSource would otherwise be rebuilt — and
# reconnected — on every single question.
_source = None
_built = False


def get_email_source():
    """Return the configured source, or None when nothing is set up."""
    global _source, _built
    if not _built:
        _source = _build_source()
        _built = True
    return _source


def reset():
    """Drop the cached source so config changes take effect (used by tests)."""
    global _source, _built
    _source = None
    _built = False


def _build_source():
    if os.environ.get("JANET_EMAIL", "").lower() == "demo":
        # The fake inbox: sample messages seeded around "now", no server needed.
        # It "sends" by recording into source.sent — nothing leaves the machine.
        return FakeEmailSource(demo_messages(datetime.now()))
    host = os.environ.get("JANET_IMAP_HOST")
    user = os.environ.get("JANET_IMAP_USER")
    password = os.environ.get("JANET_IMAP_PASSWORD")
    if host and user and password:
        port = int(os.environ.get("JANET_IMAP_PORT", "993"))   # 993 = IMAP over SSL
        # sender=None is a perfectly good mailbox: reads work, can_send() is
        # False, and the handler says "Sending isn't set up yet."
        return IMAPSource(host, user, password, port=port, sender=_get_smtp_sender())
    return None
