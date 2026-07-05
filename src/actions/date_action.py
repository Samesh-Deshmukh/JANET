from datetime import datetime


def handle(slots, ctx):
    today = datetime.now()
    return f"Today is {today:%A, %B %d, %Y}"
