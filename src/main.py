from audio.audio import record
from audio.stt import transcribe
from intent.intent import decide_action
from utils.utils import tell_time, end_timer, set_timer

print("JANET is running. Press Ctrl-C to quit.")

while True:
    audio = record()
    query = transcribe(audio)
    print(f"🗣  You said: {query.strip()}")
    action = decide_action(query)
    print(f"⚙️  Action: {action}")