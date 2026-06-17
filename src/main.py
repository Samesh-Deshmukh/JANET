from audio.audio import record
from audio.stt import transcribe
from intent.intent import decide_action
from utils.utils import tell_time, end_timer, set_timer

while True:
    file = record()
    query = transcribe(file)
    print(query)
    action = decide_action(query)
    print(action)