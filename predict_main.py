import tensorflow as tf
import numpy as np
from PIL import Image

# 1. CONFIGURATION
MODEL_PATH = 'saved_models/main_image_classifier.keras'
IMG_SIZE = (224, 224)
IMAGE_TO_CLASSIFY = 'data/main/test/other_xray/MildG2 (1).png'

# IMPORTANT: Must match the alphabetical order from training!
CLASS_NAMES = ['not_xray', 'other_xray', 'spinal_xray']

# 2. LOAD MODEL & IMAGE
model = tf.keras.models.load_model(MODEL_PATH)

full_image = Image.open(IMAGE_TO_CLASSIFY).convert('RGB')
processed_pil_image = full_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

img_array = tf.keras.utils.img_to_array(processed_pil_image)
img_batch = np.expand_dims(img_array, axis=0)

# 3. MAKE & INTERPRET PREDICTION
prediction = model.predict(img_batch)
predicted_index = np.argmax(prediction[0])
predicted_class = CLASS_NAMES[predicted_index]
confidence = prediction[0][predicted_index]

print("\nImage Classification Result")
print(f"✅ Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print("---------------------------------")