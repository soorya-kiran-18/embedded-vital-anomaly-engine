import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("apnea_detector_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("apnea_detector_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully")
print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
