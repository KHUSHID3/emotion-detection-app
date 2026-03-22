import os
import cv2
import numpy as np

# Base directory (dataset/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "archive", "train")

# Emotion label mapping
emotion_map = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "sad": 4,
    "surprised": 5,
    "neutral": 6
}

X = []  # images
y = []  # labels

print("Loading images...")

for emotion, label in emotion_map.items():
    emotion_folder = os.path.join(DATASET_PATH, emotion)

    if not os.path.exists(emotion_folder):
        continue

    for img_name in os.listdir(emotion_folder)[:300]:  # limit for safety
        img_path = os.path.join(emotion_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (48, 48))
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Images shape:", X.shape)
print("Labels shape:", y.shape)
print("Sample labels:", y[:10])
# Normalize images (0–255 -> 0–1)
X = X / 255.0

# Add channel dimension for CNN (N, 48, 48, 1)
X = X.reshape(-1, 48, 48, 1)

print("After normalization & reshape:", X.shape)
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "archive", "train")

emotion_map = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "sad": 4,
    "surprised": 5,
    "neutral": 6
}

X, y = [], []

for emotion, label in emotion_map.items():
    folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(folder):
        continue

    for img_name in os.listdir(folder)[:300]:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (48, 48))
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# 🔹 STEP-3A (already done)
X = X / 255.0
X = X.reshape(-1, 48, 48, 1)

# 🔹 STEP-3B (THIS IS WHAT YOU ASKED)
np.save(os.path.join(BASE_DIR, "X_emotion.npy"), X)
np.save(os.path.join(BASE_DIR, "y_emotion.npy"), y)

print("✅ Emotion data saved successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
import pandas as pd

# Reverse mapping
label_to_emotion = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral"
}

# Load saved labels
y_loaded = np.load(os.path.join(BASE_DIR, "y_emotion.npy"))

# Create dataframe
emotion_list = [label_to_emotion[int(label)] for label in y_loaded]

df = pd.DataFrame({
    "employee_id": range(1, len(emotion_list) + 1),
    "emotion": emotion_list
})

# Save CSV
csv_path = os.path.join(BASE_DIR, "emotion_output.csv")
df.to_csv(csv_path, index=False)

print("✅ emotion_output.csv generated")
print(df.head())
# Emotion → Stress mapping
emotion_to_stress = {
    "happy": "Low",
    "neutral": "Low",
    "surprised": "Medium",
    "sad": "Medium",
    "fearful": "High",
    "angry": "High",
    "disgusted": "High"
}

# Load emotion CSV
emotion_csv = os.path.join(BASE_DIR, "emotion_output.csv")
df = pd.read_csv(emotion_csv)

# Map stress
df["stress_level"] = df["emotion"].map(emotion_to_stress)

# Save final stress CSV
stress_csv = os.path.join(BASE_DIR, "emotion_stress_output.csv")
df.to_csv(stress_csv, index=False)

print("✅ emotion_stress_output.csv generated")
print(df.head())
