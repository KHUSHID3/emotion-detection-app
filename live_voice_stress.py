import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr

# Audio settings
DURATION = 5          # seconds
SAMPLE_RATE = 44100
AUDIO_FILE = "voice_input.wav"

print("🎤 Recording... Speak now")

# Record audio (float32)
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32'
)
sd.wait()

print("✅ Recording complete")

# 🔑 FIX: Convert float32 → int16 (PCM WAV)
audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)

# Save WAV in correct PCM format
wav.write(AUDIO_FILE, SAMPLE_RATE, audio_int16)

# Speech recognition
r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
    audio_data = r.record(source)

try:
    text = r.recognize_google(audio_data)
    print("🗣 You said:", text)

    stress_words = [
        "stress", "tired", "pressure", "exhausted",
        "overworked", "anxious", "deadline"
    ]

    if any(word in text.lower() for word in stress_words):
        print("🚨 Detected Stress Level: HIGH STRESS")
    else:
        print("🙂 Detected Stress Level: LOW STRESS")

except sr.UnknownValueError:
    print("❌ Could not understand audio")
except sr.RequestError:
    print("❌ Speech recognition service error")
