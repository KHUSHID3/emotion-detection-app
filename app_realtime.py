import streamlit as st
import cv2
import numpy as np
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI-Powered Task Optimizer", layout="centered")
st.title("🧠 AI-Powered Task Optimizer")
st.write("Image Upload | Live Webcam | Live Voice")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "emotion_cnn_model.h5"))

emotion_labels = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral"
}

emotion_to_stress = {
    "happy": "Low Stress",
    "neutral": "Low Stress",
    "surprised": "Medium Stress",
    "sad": "Medium Stress",
    "fearful": "High Stress",
    "angry": "High Stress",
    "disgusted": "High Stress"
}

def recommend_task(stress):
    if stress == "Low Stress":
        return "Assign challenging and creative tasks."
    elif stress == "Medium Stress":
        return "Assign priority tasks with manageable workload."
    else:
        return "Assign light tasks and recommend a break."

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "📹 Live Webcam", "🎤 Live Voice"])

# ========== TAB 1: IMAGE UPLOAD ==========
with tab1:
    st.subheader("📷 Image-based Emotion Detection")

    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

        st.image(img, caption="Uploaded Image", width=300)

        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = img.reshape(1, 48, 48, 1)

        pred = model.predict(img, verbose=0)
        emotion = emotion_labels[np.argmax(pred)]
        stress = emotion_to_stress[emotion]

        st.success(f"Emotion: {emotion}")
        st.info(f"Stress Level: {stress}")
        st.warning(recommend_task(stress))

# ========== TAB 2: LIVE WEBCAM ==========
with tab2:
    st.subheader("📹 Live Webcam Emotion Detection")

    if st.checkbox("Start Webcam"):
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            pred = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(pred)]
            stress = emotion_to_stress[emotion]

            cv2.putText(frame, f"{emotion}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            frame_window.image(frame, channels="BGR")

        cap.release()
        st.info(f"Stress Level: {stress}")
        st.warning(recommend_task(stress))

# ========== TAB 3: LIVE VOICE ==========
with tab3:
    st.subheader("🎤 Live Voice Stress Detection")

    if st.button("Record Voice (5 seconds)"):
        DURATION = 5
        SAMPLE_RATE = 44100
        AUDIO_FILE = "voice_input.wav"

        st.write("Recording... Speak now")

        audio = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype="float32")
        sd.wait()

        audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
        wav.write(AUDIO_FILE, SAMPLE_RATE, audio_int16)

        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio_data = r.record(source)

        try:
            text = r.recognize_google(audio_data)
            st.write("You said:", text)

            stress_words = ["stress", "tired", "pressure", "exhausted",
                            "overworked", "anxious", "deadline"]

            if any(w in text.lower() for w in stress_words):
                stress = "High Stress"
            else:
                stress = "Low Stress"

            st.success(f"Stress Level: {stress}")
            st.warning(recommend_task(stress))

        except:
            st.error("Could not understand audio")
