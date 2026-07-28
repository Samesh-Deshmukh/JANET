# src/integrations/smtp_sender.py
"""Sending mail over SMTP, using only the standard library (smtplib + email).

Its own file because it is its own concern: IMAP reads a mailbox, SMTP hands a
message to a relay. They are different servers, different ports and different
credentials, and mixing them would put the one irreversible operation in the
app inside the file whose whole point is that it never changes anything.
IMAPSource composes a sender in (see imap_source.py) rather than inheriting one.

build_reply is a pure function — Message in, EmailMessage out, no socket — so
the fiddly part (headers, threading, not double-prefixing "Re:") can be checked
offline without a mail account. Only the class needs a server.
"""
import smtplib
from email.message import EmailMessage

from integrations.email_source import EmailUnavailable, reply_subject

# Failures that mean "the relay is unreachable or said no" rather than "our code
# is wrong": OSError covers DNS/socket/TLS problems, SMTPException covers
# protocol refusals such as a rejected login or a refused recipient.
# Everything else propagates, so bugs stay visible.
_OPERATIONAL = (OSError, smtplib.SMTPException)

# Two ways to get an encrypted SMTP connection, and providers disagree on which:
#   465 — implicit SSL: the socket is encrypted from the first byte (our default).
#   587 — submission: starts in the clear, then STARTTLS upgrades it.
# We default to 465 and switch to the STARTTLS handshake when the configured
# port is 587, because some providers (Office 365) only offer that one.
STARTTLS_PORT = 587


def build_reply(original, body, from_address):
    """Build the reply to `original`. Pure: no network, no state.

    The headers are what make this a *reply* rather than a new email that
    happens to mention the same subject:
      To          — the address the original asked replies to go to.
      Subject     — "Re: " + the original, added at most once (reply_subject).
      In-Reply-To — the exact message this answers.
      References  — the thread chain; with one ancestor it's the same id.
    Without the last two, every reply starts a new thread in the recipient's
    client, which is how a human notices a machine wrote it.
    """
    reply = EmailMessage()
    reply["From"] = from_address
    reply["To"] = original.reply_to
    reply["Subject"] = reply_subject(original.subject)
    if original.message_id:
        # Omitted entirely when unknown: an empty In-Reply-To is worse than none
        # (some servers reject it), and a missing one merely loses threading.
        reply["In-Reply-To"] = original.message_id
        reply["References"] = original.message_id
    # set_content picks the charset and transfer encoding for us, so an accent
    # or an emoji in a dictated reply goes out correctly instead of as mojibake.
    reply.set_content(body)
    return reply


class SMTPSender:
    """Sends replies through an SMTP relay. Connects per send and hangs up —
    JANET sends rarely, so a pooled connection isn't worth the state."""

    def __init__(self, host, user, password, port=465, from_address=None):
        self._host = host
        self._user = user
        self._password = password
        self._port = port
        # Usually the same as the login, but some relays let you send as an
        # alias — JANET_EMAIL_FROM exists for that.
        self._from = from_address or user

    def send_reply(self, message, body):
        """Send `body` as a reply to `message`. Raises EmailUnavailable on failure."""
        reply = build_reply(message, body, self._from)
        try:
            # `with` closes the connection whatever happens, including on an
            # exception — the same guarantee imap_source gets from try/finally.
            with self._connect() as conn:
                conn.login(self._user, self._password)
                conn.send_message(reply)         # reads To/From off the headers
        except _OPERATIONAL as err:
            raise EmailUnavailable(f"could not send mail via {self._host}") from err

    def _connect(self):
        """An encrypted connection, by whichever route this port implies."""
        if self._port == STARTTLS_PORT:
            conn = smtplib.SMTP(self._host, self._port)
            conn.starttls()                      # upgrade BEFORE the password is sent
            return conn
        return smtplib.SMTP_SSL(self._host, self._port)
