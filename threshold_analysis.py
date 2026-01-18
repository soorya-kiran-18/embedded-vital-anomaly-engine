import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Load model & data
model = tf.keras.models.load_model("apnea_detector_model.keras")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Get probabilities
y_prob = model.predict(X_test).flatten()

thresholds = [0.3, 0.5, 0.7]

print("\nTHRESHOLD ANALYSIS\n")

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"Threshold = {t}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print("-" * 30)
