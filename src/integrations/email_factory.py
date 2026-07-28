# src/integrations/email_factory.py
"""Choose the configured EmailSource from env: demo, IMAP, or None.

Keeping the choice here means the handler never imports a specific backend —
adding JMAP or a local Maildir later is a change to this file only.
"""
import os
from datetime import datetime

from dotenv import load_dotenv

from integrations.email_source import FakeEmailSource
from integrations.demo_email import demo_messages
from integrations.imap_source import IMAPSource

load_dotenv()


def get_email_source():
    """Return the configured source, or None when nothing is set up."""
    if os.environ.get("JANET_EMAIL", "").lower() == "demo":
        # The fake inbox: sample messages seeded around "now", no server needed.
        return FakeEmailSource(demo_messages(datetime.now()))
    host = os.environ.get("JANET_IMAP_HOST")
    user = os.environ.get("JANET_IMAP_USER")
    password = os.environ.get("JANET_IMAP_PASSWORD")
    if host and user and password:
        port = int(os.environ.get("JANET_IMAP_PORT", "993"))   # 993 = IMAP over SSL
        return IMAPSource(host, user, password, port=port)
    return None
