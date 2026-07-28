# src/integrations/imap_source.py
"""IMAP-backed EmailSource, using only the standard library (imaplib + email).

THE READS ARE READ-ONLY, twice over: the mailbox is selected with readonly=True
and bodies are fetched with BODY.PEEK[] instead of BODY[]. A plain BODY[] fetch
sets the \\Seen flag, so merely asking "what's in my inbox?" would silently mark
your mail as read — a real, user-visible side effect we must never cause.

Sending is NOT done here. IMAP cannot send at all; that needs SMTP, a different
server on a different port. So this class *composes* an optional
smtp_sender.SMTPSender and delegates send_reply to it, which keeps the one
irreversible operation out of the file whose job is to change nothing, and lets
a mailbox be readable with no way to send (can_send() then answers False).

Parsing (_to_message) is a pure function over raw message bytes, so it can be
tested offline on a sample email; only the class needs a server.
"""
import email
import imaplib
from datetime import datetime
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime

from integrations.email_source import Message, EmailUnavailable

# Failures that mean "the mail server is unreachable or said no" rather than
# "our code is wrong": OSError covers DNS/socket/TLS problems, IMAP4.error covers
# protocol-level refusals such as a rejected login. Everything else propagates.
_OPERATIONAL = (OSError, imaplib.IMAP4.error)


def _decode(raw_header):
    """Decode a MIME-encoded header into plain text.

    Headers can only carry ASCII, so non-ASCII text is wrapped in "encoded
    words" like =?UTF-8?B?SGVsbG8=?=. decode_header splits a header into
    (bytes-or-str, charset) chunks; we decode each and glue them back together.
    """
    if not raw_header:
        return ""
    parts = []
    for chunk, charset in decode_header(raw_header):
        if isinstance(chunk, bytes):
            # errors="replace": a wrong charset should never crash the assistant
            parts.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(chunk)                  # already plain ASCII text
    return "".join(parts).strip()


def _header_date(raw_date):
    """Parse a Date header into a naive local datetime (the app-wide convention).

    Mail dates carry a timezone ("Mon, 27 Jul 2026 09:14:00 +0200"); mixing
    aware and naive datetimes blows up on comparison, so we convert to local
    time and drop the tzinfo. An unparseable/missing date sorts oldest.
    """
    if not raw_date:
        return datetime.min
    try:
        parsed = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        return datetime.min                      # malformed header, not a bug
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _snippet(msg, limit=160):
    """A short one-line preview from the plain-text body ("" if there isn't one)."""
    part = msg
    if msg.is_multipart():
        part = None
        for candidate in msg.walk():
            # skip attachments (they have a filename) — we only want the body
            if candidate.get_content_type() == "text/plain" and not candidate.get_filename():
                part = candidate
                break
        if part is None:
            return ""                            # HTML-only mail: no clean text
    if part.get_content_type() != "text/plain":
        return ""
    payload = part.get_payload(decode=True)      # decode=True undoes base64/quoted-printable
    if not payload:
        return ""
    text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    text = " ".join(text.split())                # collapse newlines/runs of spaces
    return text[:limit].strip()


def _reply_address(msg, from_address):
    """Where a reply to this message should go.

    A sender can ask for replies elsewhere with a Reply-To header (mailing
    lists and "no-reply" senders do this constantly), and RFC 5322 says that
    header wins over From when it is present. Falls back to the From address,
    and to "" when neither parses — the handler treats "" as "can't reply".
    """
    _name, address = parseaddr(_decode(msg.get("Reply-To")))
    return address or from_address


def _to_message(raw, unread=True):
    """Raw RFC-822 message bytes -> Message. Pure: no network, no state."""
    msg = email.message_from_bytes(raw)
    # Decode first, THEN split: the display name may itself be encoded, e.g.
    # "=?UTF-8?B?Sm9zw6k=?= <jose@example.com>" -> ("José", "jose@example.com").
    name, address = parseaddr(_decode(msg.get("From")))
    return Message(
        sender=name or address or "Unknown sender",
        subject=_decode(msg.get("Subject")) or "(no subject)",
        date=_header_date(msg.get("Date")),
        snippet=_snippet(msg),
        unread=unread,
        reply_to=_reply_address(msg, address),
        # Kept exactly as it appears, angle brackets included: In-Reply-To and
        # References must quote the id verbatim or the thread doesn't match.
        message_id=(msg.get("Message-ID") or "").strip(),
    )


class IMAPSource:
    """Reads an IMAP mailbox over SSL. Connects per call and always logs out —
    a voice assistant asks rarely, so a pooled connection isn't worth the state."""

    def __init__(self, host, user, password, port=993, mailbox="INBOX", sender=None):
        self._host = host
        self._user = user
        self._password = password
        self._port = port
        self._mailbox = mailbox
        # An SMTPSender, or None when sending isn't configured. Passed in by the
        # factory rather than built here, so this file never reads the
        # environment and can be constructed in a test with a stub sender.
        self._sender = sender

    def unread_count(self):
        return self._with_connection(self._count_unseen)

    def recent(self, limit=5):
        # a lambda so _with_connection can hand the open connection to the work
        return self._with_connection(lambda conn: self._fetch_recent(conn, limit))

    # --- the one write path (delegated to SMTP) ------------------------------

    def can_send(self):
        return self._sender is not None

    def send_reply(self, message, body):
        if self._sender is None:
            # The handler checks can_send() first, so this is a backstop: better
            # a friendly "couldn't send" than an AttributeError on None.
            raise EmailUnavailable("sending is not configured")
        self._sender.send_reply(message, body)

    # --- connection plumbing -------------------------------------------------

    def _with_connection(self, work):
        """Open a connection, run work(conn), and close it no matter what.

        One place owns the try/finally so both public methods can't leak a
        socket, and one place translates server trouble into EmailUnavailable so
        callers never import imaplib.
        """
        try:
            conn = self._connect()
        except _OPERATIONAL as err:
            raise EmailUnavailable(f"could not connect to {self._host}") from err
        try:
            return work(conn)
        except _OPERATIONAL as err:
            raise EmailUnavailable(f"IMAP request failed: {err}") from err
        finally:
            self._close(conn)

    def _connect(self):
        conn = imaplib.IMAP4_SSL(self._host, self._port)
        conn.login(self._user, self._password)
        # readonly=True: the server won't let this session change any flag, so
        # checking mail can't mark it read even by accident.
        conn.select(self._mailbox, readonly=True)
        return conn

    @staticmethod
    def _close(conn):
        """Best-effort teardown. Tolerates a half-open connection: the read
        already failed, so failing to hang up politely tells us nothing new."""
        try:
            conn.close()                         # deselect the mailbox
        except _OPERATIONAL:
            pass
        try:
            conn.logout()
        except _OPERATIONAL:
            pass

    # --- the actual reads ----------------------------------------------------

    @staticmethod
    def _search(conn, criterion):
        """Message ids matching an IMAP search criterion, oldest first."""
        typ, data = conn.search(None, criterion)
        if typ != "OK" or not data or not data[0]:
            return []
        return data[0].split()                   # b'1 2 3' -> [b'1', b'2', b'3']

    def _count_unseen(self, conn):
        return len(self._search(conn, "UNSEEN"))

    def _fetch_recent(self, conn, limit):
        ids = self._search(conn, "ALL")
        if not ids:
            return []
        unread_ids = set(self._search(conn, "UNSEEN"))   # to fill Message.unread
        messages = []
        for mid in reversed(ids[-limit:]):       # ids ascend by arrival: last = newest
            typ, data = conn.fetch(mid, "(BODY.PEEK[])")   # PEEK = don't set \Seen
            if typ != "OK" or not data or not isinstance(data[0], tuple):
                continue                         # odd/empty response: skip that one
            messages.append(_to_message(data[0][1], unread=mid in unread_ids))
        # arrival order usually matches the Date header, but not always — sort so
        # "newest first" is a real guarantee of the EmailSource contract.
        return sorted(messages, key=lambda m: m.date, reverse=True)
