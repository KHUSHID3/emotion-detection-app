import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model(os.path.join(BASE_DIR, "emotion_cnn_model.h5"))

# Emotion labels (same order!)
emotion_labels = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral"
}

# Image path (test ke liye dataset se hi lo)
IMAGE_PATH = os.path.join(
    BASE_DIR, "archive", "test", "happy", os.listdir(os.path.join(BASE_DIR, "archive", "test", "happy"))[0]
)

# Read image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = img / 255.0
img = img.reshape(1, 48, 48, 1)

# Predict
pred = model.predict(img)
emotion_index = np.argmax(pred)
emotion = emotion_labels[emotion_index]

print("Predicted Emotion:", emotion)
# Emotion → Stress mapping
emotion_to_stress = {
    "happy": "Low Stress",
    "neutral": "Low Stress",
    "surprised": "Medium Stress",
    "sad": "Medium Stress",
    "fearful": "High Stress",
    "angry": "High Stress",
    "disgusted": "High Stress"
}

stress_level = emotion_to_stress[emotion]
print("Detected Stress Level:", stress_level)
