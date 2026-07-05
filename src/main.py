from audio.audio import record
from audio.stt import transcribe
from audio.tts import say
from intent.dispatch import respond
from utils.context import Context


def main():
    print("JANET is running. Press Ctrl-C to quit.")
    while True:
        audio = record()
        query = transcribe(audio)
        print(f"🗣  You said: {query.strip()}")
        ctx = Context(speak=say, query=query)
        reply = respond(query, ctx)
        print(f"⚙️  Reply: {reply}")
        say(reply)


if __name__ == "__main__":
    main()
