# src/integrations/demo_email.py
"""A fake inbox seeded relative to `now`, for demos and end-to-end testing.
Selected with JANET_EMAIL=demo. Not a real mailbox — the content says so."""
from datetime import timedelta

from integrations.email_source import Message


def demo_messages(now):
    """Sample messages arriving just before `now`, so a demo always has mail.

    Three are unread and two are read, which makes the unread count (3) differ
    from the message count (5) — a good end-to-end check.
    """
    return [
        Message(
            sender="Alex Chen",
            subject="Project update",
            date=now - timedelta(minutes=20),
            snippet="Demo message. The draft is ready for your review.",
            unread=True,
        ),
        Message(
            sender="Mom",
            subject="Dinner on Sunday?",
            date=now - timedelta(hours=3),
            snippet="Demo message. Are you free around six?",
            unread=True,
        ),
        Message(
            sender="JANET Demo",
            subject="Welcome to the demo inbox",
            date=now - timedelta(hours=6),
            snippet="Demo message. Nothing here is real mail.",
            unread=True,
        ),
        Message(
            sender="Priya Raman",
            subject="Re: Invoice for June",
            date=now - timedelta(days=1),
            snippet="Demo message. Payment went out on Monday.",
            unread=False,
        ),
        Message(
            sender="Weekly Digest",
            subject="Your week in review",
            date=now - timedelta(days=2),
            snippet="Demo message. Five stories you missed.",
            unread=False,
        ),
    ]
