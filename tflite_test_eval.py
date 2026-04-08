import numpy as np
import tensorflow as tf

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

interpreter = tf.lite.Interpreter(model_path="models/apnea_detector_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

probs_normal = []
probs_abnormal = []

for i in range(len(X_test)):
    sample = np.expand_dims(X_test[i], axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prob = float(output[0][0])

    if y_test[i] == 0:
        probs_normal.append(prob)
    else:
        probs_abnormal.append(prob)

print("Normal samples → min:", min(probs_normal), "max:", max(probs_normal))
print("Abnormal samples → min:", min(probs_abnormal), "max:", max(probs_abnormal))