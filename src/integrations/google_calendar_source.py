# src/integrations/google_calendar_source.py
"""Google Calendar-backed CalendarSource (Google Calendar API v3).

Split the same way as caldav_source: the CONVERSIONS are pure functions
(`to_event`, `to_insert_body`, `to_rfc3339`) that can be tested offline against a
sample payload, and only `GoogleCalendarSource` ever touches the network.

Auth is LAZY on purpose. calendar_factory imports this module while JANET boots,
and an OAuth consent flow opens a *browser window* — doing that at import time or
in __init__ would hang the assistant on startup. Nothing here authenticates until
a method actually needs to talk to Google.

Config (a gitignored `.env`, see calendar_factory):
  JANET_GOOGLE_CREDENTIALS  path to the OAuth client-secrets JSON downloaded from
                            Google Cloud   (default ~/.config/janet/google_credentials.json)
  JANET_GOOGLE_TOKEN        where the granted token is cached, so the browser
                            consent happens once  (default ~/.config/janet/google_token.json)
"""
import os
from datetime import datetime, date, time as dtime
from pathlib import Path

import httplib2
from dotenv import load_dotenv
from google.auth.exceptions import GoogleAuthError, RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from integrations.calendar_source import Event

# Read+write, because this source implements create_event too. A read-only scope
# would make insert() fail with a 403 at write time rather than at consent time.
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# "primary" is Google's alias for the signed-in account's own calendar, so we
# never have to hard-code a calendar id.
CALENDAR_ID = "primary"

# One page is plenty for "what's on today / this week"; we don't follow
# nextPageToken. 250 is also the API's own default page size.
MAX_RESULTS = 250

DEFAULT_CREDENTIALS_PATH = "~/.config/janet/google_credentials.json"
DEFAULT_TOKEN_PATH = "~/.config/janet/google_token.json"

# Failures that mean "Google is unreachable or said no" rather than "our code is
# wrong": HttpError is any non-2xx from the API, GoogleAuthError covers a dead or
# revoked grant, HttpLib2Error covers DNS/socket/TLS trouble underneath the
# client, and FileNotFoundError means a credential file vanished (the only files
# this code path opens are the two credential files, so it can't mask anything
# else). Everything else propagates — bugs stay visible.
#
# They are re-raised as ConnectionError because that is what
# actions/calendar_action.py already catches to say "I couldn't reach your
# calendar"; this adapter speaks the calendar layer's existing failure language.
_OPERATIONAL = (HttpError, GoogleAuthError, httplib2.HttpLib2Error,
                ConnectionError, TimeoutError, FileNotFoundError)


# --- configuration -----------------------------------------------------------

def _path(name, default):
    """A path from the environment (expanding ~), or `default` when unset/empty.

    load_dotenv() is called here rather than at import time so this module works
    standalone (not only via calendar_factory) and so a test can set the
    variables before calling. It's a no-op when there is no .env file.
    """
    load_dotenv()
    return Path(os.getenv(name, "").strip() or default).expanduser()


def credentials_path():
    """Where the OAuth client-secrets JSON from Google Cloud lives."""
    return _path("JANET_GOOGLE_CREDENTIALS", DEFAULT_CREDENTIALS_PATH)


def token_path():
    """Where the granted token is cached between runs."""
    return _path("JANET_GOOGLE_TOKEN", DEFAULT_TOKEN_PATH)


def is_configured():
    """True when there is something to authenticate with, so calendar_factory can
    decide whether to build this source. Filesystem only — never the network.

    Either file is enough: the client secrets (first run, consent still needed)
    or an already-granted token (secrets no longer strictly required to refresh).
    """
    return credentials_path().is_file() or token_path().is_file()


# --- pure conversions (no network, no credentials) ---------------------------

def _as_local_naive(value):
    """Normalize an aware datetime to a naive local one — the app-wide convention
    (identical to caldav_source._as_local_naive). Mixing aware and naive
    datetimes raises on comparison, so every Event must be naive."""
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)   # to local, drop tz
    return value


def _parse_slot(slot):
    """One side of a Google event ("start" or "end") -> (naive local dt, all_day).

    Google returns exactly one of two shapes:
      timed    {"dateTime": "2026-07-28T09:00:00+05:30", "timeZone": "Asia/Kolkata"}
      all-day  {"date": "2026-07-28"}
    """
    if "dateTime" in slot:
        # fromisoformat handles both the "+05:30" offset form and a trailing "Z".
        return _as_local_naive(datetime.fromisoformat(slot["dateTime"])), False
    if "date" in slot:
        # all-day -> midnight that day, same as the CalDAV adapter does
        return datetime.combine(date.fromisoformat(slot["date"]), dtime(0, 0)), True
    raise ValueError(f"event time has neither dateTime nor date: {slot!r}")


def to_event(item):
    """A Google Calendar API event dict -> our backend-agnostic Event (pure).

    Note on all-day events: Google's end `date` is EXCLUSIVE (the day *after* the
    last day), which is already how JANET stores them — see demo_calendar's
    "Company holiday" running today 00:00 -> tomorrow 00:00. So it converts
    straight across with no +/- 1 day fudge.
    """
    start, all_day = _parse_slot(item["start"])
    end_slot = item.get("end")
    # An event with no end is a point in time; mirror caldav_source and reuse start.
    end = _parse_slot(end_slot)[0] if end_slot else start
    return Event(
        start=start,
        end=end,
        # `or` supplies the default when the field is missing OR empty: an
        # untitled Google event really does come back with no "summary" key.
        summary=item.get("summary") or "(no title)",
        all_day=all_day,
        location=item.get("location") or "",
        # Google's own id, needed to delete this occurrence later.
        uid=item.get("id") or "",
    )


def local_timezone_name():
    """The machine's IANA timezone name ("Asia/Kolkata"), or "" if we can't tell.

    Google wants an IANA name, and Python has no reverse lookup for one, so we
    read the symlink every Linux box has: /etc/localtime ->
    /usr/share/zoneinfo/Asia/Kolkata. It's a nice-to-have — the dateTime we send
    already carries a UTC offset, which is unambiguous on its own; the zone name
    additionally tells Google how to move the event across a DST boundary.
    """
    link = Path("/etc/localtime")
    if link.is_symlink():
        parts = link.resolve().parts          # (..., "zoneinfo", "Asia", "Kolkata")
        if "zoneinfo" in parts:
            return "/".join(parts[parts.index("zoneinfo") + 1:])
    return ""


def to_rfc3339(dt):
    """A naive local datetime -> the RFC3339 string the API wants.

    astimezone() on a NAIVE datetime assumes it is local time and attaches this
    machine's offset, turning 09:00 into "2026-07-28T09:00:00+05:30". Sending a
    bare "09:00" would leave Google guessing which zone we meant.
    """
    return dt.astimezone().isoformat()


def to_insert_body(event, time_zone=None):
    """An Event -> the body dict events().insert wants (pure).

    `time_zone` is injectable so tests don't depend on the machine's zone;
    normally it is discovered by local_timezone_name().
    """
    body = {"summary": event.summary}
    if event.all_day:
        # All-day events use plain dates. Google treats `end` as exclusive, which
        # is exactly what our Event already holds (see to_event).
        body["start"] = {"date": event.start.date().isoformat()}
        body["end"] = {"date": event.end.date().isoformat()}
    else:
        zone = local_timezone_name() if time_zone is None else time_zone
        body["start"] = {"dateTime": to_rfc3339(event.start)}
        body["end"] = {"dateTime": to_rfc3339(event.end)}
        if zone:
            # Omitted rather than sent empty: the offset in dateTime already
            # pins the moment, and Google rejects a blank timeZone.
            body["start"]["timeZone"] = zone
            body["end"]["timeZone"] = zone
    if event.location:
        body["location"] = event.location
    return body


# --- OAuth (lazy: nothing below runs until an API call needs it) -------------

def _cached_credentials(token_file):
    """Load the token saved by a previous run, or None on the very first run."""
    if not token_file.is_file():
        return None
    return Credentials.from_authorized_user_file(str(token_file), SCOPES)


def _refreshed(creds):
    """Renew an expired access token using the stored refresh token — silent, no
    browser. Returns None if Google refused (the grant was revoked or expired),
    so the caller falls back to asking for consent again."""
    try:
        creds.refresh(Request())
    except RefreshError:
        return None
    return creds


def _consent():
    """The one-time interactive flow: opens a browser, user approves, we get a
    token. run_local_server(port=0) spins up a throwaway localhost server on a
    free port to catch Google's redirect — the standard "installed app" dance."""
    secrets = credentials_path()
    if not secrets.is_file():
        raise FileNotFoundError(
            f"Google client secrets not found at {secrets}. Download the OAuth "
            f"client JSON from Google Cloud and put it there, or set "
            f"JANET_GOOGLE_CREDENTIALS."
        )
    print("🔐 Authorising JANET with Google — a browser window will open (one time).")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    return flow.run_local_server(port=0)


def _save_token(creds, token_file):
    """Persist the token so consent happens exactly once.

    This file IS a credential — anyone who reads it can read and write the
    calendar — so the directory and file are owner-only. touch(mode=0o600) before
    writing means the secret is never briefly world-readable; the chmod after it
    tightens a file that already existed with looser permissions.
    """
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.parent.chmod(0o700)
    token_file.touch(mode=0o600, exist_ok=True)
    token_file.chmod(0o600)
    token_file.write_text(creds.to_json())


def load_credentials():
    """Usable OAuth credentials: cached, refreshed, or freshly consented.

    Called only from GoogleCalendarSource._api(), i.e. only when a request is
    about to happen — never at import or construction time.
    """
    token_file = token_path()
    creds = _cached_credentials(token_file)
    if creds and creds.valid:
        return creds                       # the common case: nothing to do
    if creds and creds.expired and creds.refresh_token:
        creds = _refreshed(creds)          # silent renewal
    if creds is None or not creds.valid:
        creds = _consent()                 # first run, or the grant was revoked
    _save_token(creds, token_file)
    return creds


def authorize():
    """Run the consent flow now, from a terminal, instead of mid-conversation.

        cd src && ../venv/bin/python -c \
            "from integrations.google_calendar_source import authorize; authorize()"

    Worth doing once before starting JANET: the browser flow blocks, and JANET's
    main loop is single-threaded, so it would freeze the mic while you click.
    """
    load_credentials()
    print(f"✅ Google Calendar authorised. Token saved to {token_path()}")


# --- the live source ---------------------------------------------------------

class GoogleCalendarSource:
    """A CalendarSource backed by Google Calendar API v3.

    Construction is free and side-effect-free: no file is read, no network call
    is made, no browser opens. The service (and therefore the OAuth flow) is
    built on the first real request and reused after that.
    """

    def __init__(self, calendar_id=CALENDAR_ID):
        self._calendar_id = calendar_id
        self._service = None               # built lazily, then cached

    def events_between(self, start, end):
        response = self._execute(lambda api: api.list(
            calendarId=self._calendar_id,
            timeMin=to_rfc3339(start),
            timeMax=to_rfc3339(end),
            # Expand recurring events into individual occurrences — the same job
            # as the CalDAV adapter's expand=True. Also required by orderBy.
            singleEvents=True,
            orderBy="startTime",
            maxResults=MAX_RESULTS,
        ))
        items = response.get("items", [])
        # A cancelled occurrence of a recurring event can come back with no
        # start/end, which would blow up to_event. Skip those.
        events = [to_event(item) for item in items if item.get("status") != "cancelled"]
        # Google already sorts by start time; sorting again makes the
        # CalendarSource contract true regardless of what the server did.
        return sorted(events, key=lambda e: e.start)

    def create_event(self, event):
        self._execute(lambda api: api.insert(
            calendarId=self._calendar_id,
            body=to_insert_body(event),
        ))

    def delete_event(self, event):
        """Delete by Google's own event id.

        Refuses without an id rather than searching for something that looks
        similar: this is reached from a spoken sentence that may have been
        misheard, and a deletion cannot be undone by saying "no" afterwards.
        """
        if not event.uid:
            raise LookupError("no id for that event")
        self._execute(lambda api: api.delete(
            calendarId=self._calendar_id,
            eventId=event.uid,
        ))

    # --- plumbing ------------------------------------------------------------

    def _api(self):
        """The events() collection, authorizing and building the service once."""
        if self._service is None:
            # cache_discovery=False: the default on-disk discovery cache needs
            # the long-obsolete oauth2client and just prints a warning.
            self._service = build("calendar", "v3",
                                  credentials=load_credentials(),
                                  cache_discovery=False)
        return self._service.events()

    def _execute(self, build_request):
        """Authorize (once), run one API request, and translate backend trouble
        into ConnectionError. One place owns the translation so both public
        methods stay readable — same shape as imap_source._with_connection."""
        try:
            return build_request(self._api()).execute()
        except _OPERATIONAL as err:
            raise ConnectionError(f"Google Calendar request failed: {err}") from err
