from utils.utils import tell_time, end_timer, set_timer

def decide_action(query):
    if "what is the time" in query.lower():
        tell_time()
        return "time"
    elif "timer" in query.lower():
        set_timer()
        return "timer"
    else:
        return ""