from datetime import datetime


def handle(slots, ctx):
    # Computed at call time — the old tell_time() froze this at import.
    now = datetime.now()
    return f"The time is {now:%I:%M %p}"
