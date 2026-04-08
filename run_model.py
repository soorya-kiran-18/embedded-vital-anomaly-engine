import numpy as np
import tensorflow as tf

print("SCRIPT STARTED")

# ---------- NORMALIZATION CONSTANTS ----------
# These should ideally come from your training dataset
HR_MEAN, HR_STD = 75.2, 8.4
RR_MEAN, RR_STD = 16.0, 3.0
SPO2_MEAN, SPO2_STD = 97.0, 1.5

# ---------- LOAD TFLITE MODEL ----------
interpreter = tf.lite.Interpreter(model_path="apnea_detector_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Enter 30 rows of: HR RR SPO2 MOV")

window = []

# ---------- INPUT ----------
for i in range(30):
    values = list(map(float, input(f"Timestep {i+1}: ").split()))
    window.append(values)

window = np.array(window, dtype=np.float32)

# ---------- NORMALIZATION ----------
# Only normalize HR, RR, SpO2
window[:,0] = (window[:,0] - HR_MEAN) / HR_STD
window[:,1] = (window[:,1] - RR_MEAN) / RR_STD
window[:,2] = (window[:,2] - SPO2_MEAN) / SPO2_STD

# Movement stays unchanged

# ---------- SHAPE FIX ----------
window = np.expand_dims(window, axis=0)

print("Input shape:", window.shape)

# ---------- INFERENCE ----------
interpreter.set_tensor(input_details[0]['index'], window)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

probability = float(output[0][0])

print("\nApnea Probability:", probability)

# ---------- RESULT ----------
if probability > 0.5:
    print("Result: ABNORMAL PATTERN DETECTED")
else:
    print("Result: NORMAL PATTERN")