import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
DATA_DIR = r"ASL_Data"
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26
EPOCHS = 30
BATCH_SIZE = 32

# NORMALIZATION FUNCTION
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]              # landmark 0 = wrist
    landmarks = landmarks - wrist
    return landmarks.flatten()

# LOAD DATA
data = []
labels = []

for label in CLASS_NAMES:
    folder = os.path.join(DATA_DIR, label)
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                raw = np.load(os.path.join(folder, file))
                normalized = normalize_landmarks(raw)
                data.append(normalized)
                labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Total samples:", data.shape[0])
print("Feature size:", data.shape[1])

# LABEL ENCODING
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded, NUM_CLASSES)
# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels_onehot,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# MODEL (MLP)
model = Sequential([
    Dense(256, activation='relu', input_shape=(data.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# TRAIN MODEL
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# SAVE MODEL
model.save("asl_landmark_dl_model.h5")
print("Model saved as asl_landmark_dl_model.h5")

# PLOT ACCURACY CURVE

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()

# MODEL EVALUATION

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print("\nOverall Test Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=CLASS_NAMES
))


# CONFUSION MATRIX
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for ASL Alphabet Classification")
plt.tight_layout()
plt.show()
