import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import json
from datetime import datetime
from scipy.spatial.distance import cosine
import io
import random

# Configure page
st.set_page_config(
    page_title="Comprehensive Biometric",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# UI CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e0f7ff 0%, #cce7ff 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #0077b6 0%, #0096c7 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = 'fingerprint_model2.h5'
DATA_FOLDER = 'Data'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@st.cache_resource
def load_fingerprint_model():
    model = load_model(MODEL_PATH)
    return Model(inputs=model.input, outputs=model.layers[-2].output)

feature_extractor = load_fingerprint_model()

def preprocess_image(image):
    img = Image.open(image) if isinstance(image, str) else image
    img = img.convert('L').convert('RGB').resize((224, 224))
    img_array = np.expand_dims(img_to_array(img), axis=0).astype('float32') / 255.0
    return img_array

def extract_features(img_array):
    with st.spinner('Analyzing fingerprint...'):
        features = feature_extractor.predict(img_array, verbose=0)
        return features / np.linalg.norm(features)

def compare_fingerprints(uploaded_features):
    best_match = None
    best_similarity = 0
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(('.BMP', '.bmp')):
            img_array = preprocess_image(os.path.join(DATA_FOLDER, file))
            features = extract_features(img_array)
            similarity = 1 - cosine(uploaded_features.flatten(), features.flatten())
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = file
    return best_match, best_similarity

def generate_person_info(name):
    return {
        "full_name": name,
        "age": random.randint(19, 21),
        "place": random.choice(["Hyderabad", "Telanagana"]),
        "date_of_birth": f"{random.randint(1950, 1970)}-01-01",
        "national_id_number": f"{random.randint(10000000, 99999999)}",
        "occupation": "Student",
        "years_of_service": random.randint(25, 35),
        "monthly_pension_amount": f"{random.randint(30000, 80000):,}",
        "bank_account_number": f"{random.randint(100000000000, 999999999999)}",
        "contact_number": f"+254 {random.randint(700000000, 799999999)}"
    }

def main():
    st.markdown('<div class="main-header"><h1>Comprehensive Biometric</h1></div>', unsafe_allow_html=True)

    st.sidebar.title("Add Fingerprint")
    with st.sidebar.form("add_form"):
        new_name = st.text_input("Person's Name")
        new_image = st.file_uploader("Fingerprint Image", type=["bmp", "jpg", "png"])
        submitted = st.form_submit_button("Add to Database")
        if submitted and new_name and new_image:
            save_path = os.path.join(DATA_FOLDER, f"{new_name}.BMP")
            with open(save_path, "wb") as f:
                f.write(new_image.getbuffer())
            st.success(f"Added {new_name} to the fingerprint database")

    st.markdown("### 🔍 Upload Fingerprint")
    uploaded_file = st.file_uploader("Choose a fingerprint", type=['bmp', 'png', 'jpg'])

    if uploaded_file:
        img_array = preprocess_image(Image.open(uploaded_file))
        uploaded_features = extract_features(img_array)
        best_match, similarity = compare_fingerprints(uploaded_features)

        if best_match and similarity > 0.01:
            matched_name = os.path.splitext(best_match)[0]
            st.image(uploaded_file, caption="Uploaded Fingerprint", width=300)
            st.success(f"Matched with: {matched_name} ({int(similarity * 100)}% confidence)")
            person_info = generate_person_info(matched_name)
            for key, value in person_info.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.error("No matching fingerprint found.")

if __name__ == "__main__":
    main()
