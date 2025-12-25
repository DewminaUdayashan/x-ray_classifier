import io
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import numpy as np
import tensorflow as tf

# App Initialization
app = FastAPI(
    title="Scoliosis X-Ray Classification API",
    description="Provides three endpoints to classify X-ray images.",
    version="1.0.0",
)

# Model Loading
# Load all three models into memory when the application starts.
# This is efficient as they aren't reloaded for every request.
try:
    print("Loading models...")
    MODEL_IMG_TYPE = tf.keras.models.load_model('saved_models/main_image_classifier.keras')
    MODEL_SCOLIOSIS_COND = tf.keras.models.load_model('saved_models/scoliosis_classifier.keras')
    MODEL_SCOLIOSIS_CURVE = tf.keras.models.load_model('saved_models/scoliosis_curve_classifier.keras')
    print("All models loaded successfully!")
except Exception as e:
    print(f"FATAL: Could not load models. Error: {e}")

# Configuration & Helper Functions
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes, resize_only: bool = False):
    """
    Preprocesses an image from bytes: opens, converts to RGB, crops/resizes,
    and prepares it for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    if not resize_only:
        # Crop the central vertical third for scoliosis-specific models
        width, height = img.size
        crop_width = width // 3
        left = (width - crop_width) // 2
        right = left + crop_width
        img = img.crop((left, 0, right, height))

    # Resize to the model's expected input size and convert to a NumPy array
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = tf.keras.utils.img_to_array(img)

    # Return a batch of 1 image
    return np.expand_dims(img_array, axis=0)


# API Endpoints

@app.get("/", summary="Root endpoint", description="A simple hello world to check if the API is running.")
async def root():
    return {"message": "Welcome to the X-Ray Classification API!"}


@app.post("/classify_image_type", summary="Classify Image: Spinal X-Ray, Other X-Ray, or Not an X-Ray")
async def classify_image_type(
        file: UploadFile = File(...),
        enable_cropping: bool = Form(True, description="Set to `false` to disable central spine cropping.")
):
    """
    Endpoint 1: Validates the input image.
    - **Input**: Any image file.
    - **Parameter**: `enable_cropping` (boolean, default: True).
    - **Output**: Classification as 'spinal_xray', 'other_xray', or 'not_xray'.
    """
    image_bytes = await file.read()
    # For this model, we use the full image without cropping
    processed_image = preprocess_image(image_bytes, resize_only=not enable_cropping)

    prediction = MODEL_IMG_TYPE.predict(processed_image)

    class_names = ['not_xray', 'other_xray', 'spinal_xray']
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])

    return {"predicted_class": predicted_class, "confidence": f"{confidence:.2%}"}


@app.post("/check_scoliosis_condition", summary="Check for Scoliosis Condition (Normal vs. Scoliosis)")
async def check_scoliosis_condition(
        file: UploadFile = File(...),
        enable_cropping: bool = Form(True, description="Set to `false` to disable central spine cropping.")
):
    """
    Endpoint 2: Checks if a spinal X-ray shows signs of scoliosis.
    - **Input**: A spinal X-ray image.
    - **Parameter**: `enable_cropping` (boolean, default: True).
    - **Output**: Classification as 'normal' or 'scoliosis'.
    """
    image_bytes = await file.read()

    processed_image = preprocess_image(image_bytes, resize_only=not enable_cropping)

    prediction = MODEL_SCOLIOSIS_COND.predict(processed_image)
    score = float(prediction[0][0])

    if score > 0.5:
        return {"predicted_class": "scoliosis", "confidence": f"{score:.2%}"}
    else:
        return {"predicted_class": "normal", "confidence": f"{1 - score:.2%}"}


@app.post("/classify_scoliosis_curve", summary="Classify Scoliosis Curve Type (C-Curve vs. S-Curve)")
async def classify_scoliosis_curve(
        file: UploadFile = File(...),
        enable_cropping: bool = Form(True, description="Set to `false` to disable central spine cropping.")
):
    """
    Endpoint 3: Classifies the type of scoliosis curve.
    - **Input**: A scoliosis positive spinal X-ray image.
    - **Parameter**: `enable_cropping` (boolean, default: True).
    - **Output**: Classification as 'C-Curve' or 'S-Curve'.
    """
    image_bytes = await file.read()

    processed_image = preprocess_image(image_bytes, resize_only=not enable_cropping)

    prediction = MODEL_SCOLIOSIS_CURVE.predict(processed_image)
    score = float(prediction[0][0])

    class_names = ['C-Curve', 'S-Curve']  # Matches training: ['c_type', 's_type']
    if score > 0.5:
        return {"predicted_class": class_names[1], "confidence": f"{score:.2%}"}
    else:
        return {"predicted_class": class_names[0], "confidence": f"{1 - score:.2%}"}

