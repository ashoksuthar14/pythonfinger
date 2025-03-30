import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.repeat(img, 3, axis=-1)  # Convert to 3 channels
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to match predicted label to Subject ID
def get_matched_subject(pred):
    predicted_label = np.argmax(pred, axis=1)[0]
    return f"Subject ID: {predicted_label}"
