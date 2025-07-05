import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('data/sign_language_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('data/sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)