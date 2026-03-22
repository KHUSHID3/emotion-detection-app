import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

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

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)

    pred = model.predict(face, verbose=0)
    emotion = emotion_labels[np.argmax(pred)]

    cv2.putText(frame, f"Emotion: {emotion}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Live Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
