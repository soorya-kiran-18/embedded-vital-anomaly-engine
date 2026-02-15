import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# LOAD PREPARED DATA
# -----------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# -----------------------------
# MODEL DEFINITION
# -----------------------------
model = Sequential([
    Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(30, 4)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=32, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),

    Dense(16, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# PLOT LOSS CURVE (FILE-ONLY BACKEND)
# -----------------------------
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend
import matplotlib.pyplot as plt

print("Plotting loss curve...")

plt.figure(figsize=(8,5))

epochs = range(1, len(history.history['loss']) + 1)

plt.plot(epochs, history.history['loss'], label='train_loss')
plt.plot(epochs, history.history['val_loss'], label='val_loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence (Loss Curves)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)

print("Loss curve saved as loss_curve.png")

# -----------------------------
# EVALUATE ON TEST SET
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("apnea_detector_model.keras")

print("Model saved as apnea_detector_model.keras")
