import os
import numpy as np
import cv2
import io
import keras

# Load model once at startup
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'model.keras')
model = keras.saving.load_model(model_path)

def predict_image(img_bytes):
    # Step 1: Decode image bytes → numpy array (same as cv2.imread)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Step 2: Resize to 100x100 (your model's input size)
    img   = cv2.resize(img, (100, 100))

    # Step 3: Normalize 0-255 → 0-1
    img   = img / 255.0

    # Step 4: Reshape (1, 100, 100, 3)
    img   = img.reshape(1, 100, 100, 3)

    # Step 5: Predict raw probability
    raw_pred = float(model.predict(img, verbose=0)[0][0])

    # Step 6: Apply threshold > 0.5
    is_cat = raw_pred > 0.5

    # Step 7: Get label & confidence
    if is_cat:
        label      = "Cat 🐱"
        confidence = raw_pred * 100
    else:
        label      = "Dog 🐶"
        confidence = (1 - raw_pred) * 100

    return {
        "class"     : label,
        "confidence": round(confidence, 2),
        "raw_prob"  : round(raw_pred, 4)
    }