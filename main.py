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
    page_title="Fingerprint Recognition System",
    page_icon="👆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with improved design
st.markdown("""
<style>
    /* Main container styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
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
    
    /* Upload section styles */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
        border: 2px dashed #1e3c72;
    }
    
    /* Results section styles */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .score-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: conic-gradient(#4CAF50 var(--percentage), #f3f3f3 var(--percentage));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        font-size: 36px;
        font-weight: bold;
        color: #1e3c72;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: relative;
        transition: all 0.3s ease;
    }
    
    .score-circle::after {
        content: '';
        position: absolute;
        width: 190px;
        height: 190px;
        border-radius: 50%;
        background: white;
        z-index: 0;
    }
    
    .score-value {
        position: relative;
        z-index: 1;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Person info styles */
    .person-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #1e3c72;
    }
    
    .info-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .info-label {
        color: #6c757d;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .info-value {
        color: #1e3c72;
        font-weight: 600;
        font-size: 1.2em;
    }
    
    /* Image comparison styles */
    .image-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
    }
    
    .image-container img {
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Loading animation */
    .stSpinner {
        border-color: #1e3c72 !important;
    }
    
    /* Custom button styles */
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improved section headers */
    h3 {
        color: #1e3c72;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e4e8eb;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize configurations
MODEL_PATH = 'fingerprint_model2.h5'
DATA_FOLDER = 'data'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
@st.cache_resource
def load_fingerprint_model():
    model = load_model(MODEL_PATH)
    return Model(inputs=model.input, outputs=model.layers[-2].output)

feature_extractor = load_fingerprint_model()

def preprocess_image(image):
    """Preprocess image for the model."""
    img = Image.open(image) if isinstance(image, str) else image
    # Convert to grayscale first for better feature extraction
    img = img.convert('L')
    # Convert back to RGB (3 channels) as required by the model
    img = img.convert('RGB')
    # Resize to model's expected size
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = img_to_array(img)
    # Add batch dimension and normalize
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def extract_features(img_array):
    """Extract features from preprocessed image."""
    with st.spinner('Analyzing fingerprint...'):
        features = feature_extractor.predict(img_array, verbose=0)
        # Normalize the features
        features = features / np.linalg.norm(features)
        return features

def compare_fingerprints(uploaded_features):
    """Compare fingerprint features with database."""
    best_match = None
    best_similarity = 0
    
    with st.spinner('Searching for matches...'):
        for file in os.listdir(DATA_FOLDER):
            if file.endswith(('.BMP', '.bmp')):
                file_path = os.path.join(DATA_FOLDER, file)
                img_array = preprocess_image(file_path)
                features = extract_features(img_array)
                
                # Calculate cosine similarity
                similarity = 1 - cosine(uploaded_features.flatten(), features.flatten())
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = file
    
    return best_match, best_similarity

def generate_random_person_info():
    """Generate random person information."""
    kenyan_names = [
        "Kamau Njoroge", "Wanjiku Muthoni", "Ochieng Otieno", "Akinyi Atieno",
        "Kipchoge Keino", "Wangari Maathai", "Mwai Kibaki", "Ngũgĩ wa Thiong'o"
    ]
    
    cities = [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika",
        "Malindi", "Kitale", "Garissa", "Nyeri"
    ]
    
    occupations = [
        "School Teacher", "Civil Servant", "Bank Manager", "Police Officer",
        "Agricultural Officer", "Hospital Administrator", "Railway Engineer",
        "Post Office Manager", "University Lecturer", "Government Administrator"
    ]
    
    # Generate random data
    age = random.randint(55, 75)
    years_service = random.randint(25, 35)
    pension_amount = random.randint(30000, 80000)
    
    return {
        "full_name": random.choice(kenyan_names),
        "age": age,
        "place": random.choice(cities),
        "date_of_birth": f"{2025-age}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "national_id_number": f"{random.randint(10000000, 99999999)}",
        "occupation": random.choice(occupations),
        "years_of_service": years_service,
        "monthly_pension_amount": f"{pension_amount:,}",
        "bank_account_number": f"{random.randint(100000000000, 999999999999)}",
        "contact_number": f"+254 {random.randint(700000000, 799999999)}"
    }

def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔍 Fingerprint Recognition System")
    st.markdown("<h2 style='font-size: 1.5rem; opacity: 0.8;'>Secure Pension Information Management</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 👆 Upload Fingerprint")
        uploaded_file = st.file_uploader(
            "Drop your fingerprint image here",
            type=['bmp', 'BMP', 'png', 'jpg', 'jpeg'],
            help="Supported formats: BMP, PNG, JPG"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Create columns for image display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Uploaded Fingerprint")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(uploaded_file, width=400, caption="Uploaded Image")
            st.markdown('</div>', unsafe_allow_html=True)

        # Process the image
        img_array = preprocess_image(Image.open(uploaded_file))
        uploaded_features = extract_features(img_array)
        best_match, similarity = compare_fingerprints(uploaded_features)

        # Lower the threshold to 0.01 (1%) for more lenient matching
        if best_match and similarity > 0.01:
            with col2:
                st.markdown("### Matched Fingerprint")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                matched_image = Image.open(os.path.join(DATA_FOLDER, best_match))
                st.image(matched_image, width=400, caption="Matched Image")
                st.markdown('</div>', unsafe_allow_html=True)

            # Match score with circular progress
            score_percentage = int(similarity * 100)
            st.markdown(f"""
                <div class="score-circle" style="--percentage: {score_percentage}%">
                    <div class="score-value">{score_percentage}%</div>
                </div>
                <p style="text-align: center; color: #1e3c72; font-size: 1.2rem; font-weight: 500;">Match Confidence</p>
            """, unsafe_allow_html=True)

            # Generate and display person information
            person_info = generate_random_person_info()
            
            st.markdown("### Person Information")
            
            # Create a grid of information items
            info_items = [
                ("👤 Full Name", person_info["full_name"]),
                ("🎂 Age", f"{person_info['age']} years"),
                ("📍 Location", person_info["place"]),
                ("📅 Date of Birth", person_info["date_of_birth"]),
                ("🆔 National ID", person_info["national_id_number"]),
                ("💼 Previous Occupation", person_info["occupation"]),
                ("⏳ Years of Service", f"{person_info['years_of_service']} years"),
                ("💰 Monthly Pension", f"KES {person_info['monthly_pension_amount']}"),
                ("🏦 Bank Account", person_info["bank_account_number"]),
                ("📞 Contact", person_info["contact_number"])
            ]
            
            # Create columns for the grid layout
            cols = st.columns(2)
            for i, (label, value) in enumerate(info_items):
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="info-box">
                            <div class="info-label">{label}</div>
                            <div class="info-value">{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("⚠️ No matching fingerprint found in the database.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
