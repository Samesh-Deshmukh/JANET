import pyaudio
import wave
import keyboard
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record():

    while not keyboard.is_pressed('space'):
        pass

    with wave.open('output.wav', 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print("Recording...")
        while keyboard.is_pressed('space'):
            wf.writeframes(stream.read(CHUNK))
        print('Recording Stopped')
        stream.close()
        p.terminate()

    return 'output.wav'