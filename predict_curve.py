import tensorflow as tf
import numpy as np
from PIL import Image

# 1. CONFIGURATION
MODEL_PATH = 'saved_models/scoliosis_curve_classifier.keras'
IMG_SIZE = (224, 224)
IMAGE_TO_CLASSIFY = 'data/test/s2.png'

# Must match the order from training ['c_type', 's_type']
CLASS_NAMES = ['C-Curve', 'S-Curve']

ENABLE_CROPPING = True

# 2. PREPROCESSING FUNCTION
def preprocess_and_crop_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    crop_width = width // 3
    left = (width - crop_width) // 2
    right = left + crop_width
    cropped_img = img.crop((left, 0, right, height))
    resized_img = cropped_img.resize(target_size, Image.Resampling.LANCZOS)
    return resized_img

# 3. LOAD MODEL & IMAGE
model = tf.keras.models.load_model(MODEL_PATH)

if ENABLE_CROPPING:
    processed_pil_image = preprocess_and_crop_image(IMAGE_TO_CLASSIFY, target_size=IMG_SIZE)
else:
    full_image = Image.open(IMAGE_TO_CLASSIFY).convert('RGB')
    processed_pil_image = full_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

img_array = tf.keras.utils.img_to_array(processed_pil_image)
img_batch = np.expand_dims(img_array, axis=0)

# 4. MAKE & INTERPRET PREDICTION
prediction = model.predict(img_batch)
score = prediction[0][0]

if score > 0.5:
    predicted_class = CLASS_NAMES[1] # S-Curve
    confidence = score
else:
    predicted_class = CLASS_NAMES[0] # C-Curve
    confidence = 1 - score

print("\nCurve Prediction Result")
print(f"✅ Predicted Curve Type: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print("-------------------------------")