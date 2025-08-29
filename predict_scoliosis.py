import tensorflow as tf
import numpy as np
import os
from PIL import Image

# --- 1. CONFIGURATION ---

# Path to your saved model
MODEL_PATH = 'saved_models/scoliosis_classifier.keras'

# Image size must be the same as during training
IMG_SIZE = (224, 224)

# The names of your classes
CLASS_NAMES = ['normal', 'scoliosis']

# Path to the new image you want to classify
IMAGE_TO_CLASSIFY = 'data/test/normal.jpg'

# --- NEW: Preprocessing Toggle ---
# Set to True to crop the image's central spine region before classification.
# Set to False to classify the full, un-cropped image.
ENABLE_CROPPING = False


# --- 2. IMAGE PREPROCESSING FUNCTION ---

def preprocess_and_crop_image(image_path, target_size=(224, 224)):
    """
    Loads a full-view X-ray, crops the central vertical third to isolate the
    spine, and then resizes it to the model's required input size.
    """
    img = Image.open(image_path).convert('RGB')

    # --- Cropping Logic ---
    width, height = img.size
    crop_width = width // 3
    left = (width - crop_width) // 2
    right = left + crop_width
    cropped_img = img.crop((left, 0, right, height))

    # --- Resizing Logic ---
    resized_img = cropped_img.resize(target_size, Image.Resampling.LANCZOS)
    return resized_img


# --- 3. LOAD THE TRAINED MODEL ---

print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 4. LOAD AND PREPARE THE IMAGE (with toggle) ---

print(f"Loading image: {IMAGE_TO_CLASSIFY}")

try:
    if ENABLE_CROPPING:
        print("Preprocessing enabled: Cropping central spine region.")
        processed_pil_image = preprocess_and_crop_image(IMAGE_TO_CLASSIFY, target_size=IMG_SIZE)
    else:
        print("Preprocessing disabled: Using full image.")
        # Load the image directly and resize it without cropping
        full_image = Image.open(IMAGE_TO_CLASSIFY).convert('RGB')
        processed_pil_image = full_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    # Convert the processed PIL image to a NumPy array
    img_array = tf.keras.utils.img_to_array(processed_pil_image)

    # The model expects a "batch" of images, so we add an extra dimension
    img_batch = np.expand_dims(img_array, axis=0)

except FileNotFoundError:
    print(f"Error: Image file not found at {IMAGE_TO_CLASSIFY}")
    exit()

# --- 5. MAKE A PREDICTION ---
print("Classifying the image...")
prediction = model.predict(img_batch)
score = prediction[0][0]

# --- 6. INTERPRET THE RESULT ---
if score > 0.5:
    predicted_class = CLASS_NAMES[1]  # scoliosis
    confidence = score
else:
    predicted_class = CLASS_NAMES[0]  # normal
    confidence = 1 - score

print("\n--- Prediction Result ---")
print(f"✅ Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print(f"(Raw model output score: {score:.4f})")
print("-------------------------")