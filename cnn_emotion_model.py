import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved data
X = np.load(os.path.join(BASE_DIR, "X_emotion.npy"))
y = np.load(os.path.join(BASE_DIR, "y_emotion.npy"))

print("Data loaded:", X.shape, y.shape)

# One-hot encode labels
y = to_categorical(y, num_classes=7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save model
model.save(os.path.join(BASE_DIR, "emotion_cnn_model.h5"))
print("✅ CNN model saved as emotion_cnn_model.h5")
