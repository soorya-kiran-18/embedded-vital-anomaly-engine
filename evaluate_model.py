import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# LOAD MODEL & TEST DATA
# -----------------------------
model = tf.keras.models.load_model("apnea_detector_model.keras")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# -----------------------------
# PREDICTIONS
# -----------------------------
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX")
print(cm)

print("\nDETAILS")
print(f"True Positives  (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives  (TN): {tn}")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, digits=4))
