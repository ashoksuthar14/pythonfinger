from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
import glob
import shutil
import google.generativeai as genai
import json
from datetime import datetime, timedelta

# ✅ Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('mysite','static', 'uploads')
app.config['MODEL_PATH'] = 'fingerprint_model2.h5'
app.config['DATA_FOLDER'] = 'data'

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCaomXYLvpJbiZUCspt56Fs8L07LVHS__0"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# ✅ Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load the trained model and create feature extractor
fingerprint_model = load_model(app.config['MODEL_PATH'])
# Create a feature extractor model (removing the last layer)
feature_extractor = Model(inputs=fingerprint_model.input, outputs=fingerprint_model.layers[-2].output)

# ✅ Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'BMP'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    """Preprocess image for model prediction."""
    IMG_SIZE = (224, 224)  # Same size used during training
    img = load_img(filepath, target_size=IMG_SIZE)  # Load image
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def extract_features(image_array):
    """Extract features from the image using the feature extractor."""
    features = feature_extractor.predict(image_array)
    return features

def compare_fingerprints(uploaded_features):
    """Compare uploaded fingerprint with all fingerprints in the data folder."""
    best_match = None
    best_similarity = -1
    best_match_path = None
    
    # Get all fingerprint images from data folder
    data_files = glob.glob(os.path.join(app.config['DATA_FOLDER'], '*.[Bb][Mm][Pp]'))
    
    for fingerprint_file in data_files:
        # Process database fingerprint
        db_img_array = preprocess_image(fingerprint_file)
        db_features = extract_features(db_img_array)
        
        # Calculate similarity (using cosine similarity)
        similarity = 1 - cosine(uploaded_features.flatten(), db_features.flatten())
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = os.path.basename(fingerprint_file)
            best_match_path = fingerprint_file
    
    # Copy the best match to static folder for display
    if best_match_path:
        static_match_path = os.path.join('static', 'matches', best_match)
        os.makedirs(os.path.join('static', 'matches'), exist_ok=True)
        shutil.copy2(best_match_path, static_match_path)
    
    return best_match, best_similarity

def generate_person_info():
    """Generate random person information using Gemini API."""
    prompt = """Generate a  person's information for a pension system in Kenya. Include:
    1. Full Name : Akash or Lithin
    2. Age : 20
    3. Place :India
    4. Department : AI
    
    Format as JSON. Use realistic Kenyan details."""

    try:
        response = gemini_model.generate_content(prompt)
        # Extract JSON from the response
        info_text = response.text
        # Find JSON content between ```json and ``` if present
        if "```json" in info_text:
            info_text = info_text.split("```json")[1].split("```")[0]
        return json.loads(info_text)
    except Exception as e:
        print(f"Error generating person info: {str(e)}")
        # Fallback basic info if JSON parsing fails
        return {
            "full_name": "John Doe",
            "age": 65,
            "place": "Nairobi, Kenya",
            "error": "Failed to generate complete information"
        }

# ✅ Home Route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    # ✅ Save file securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ✅ Process & Extract Features
    img_array = preprocess_image(filepath)
    uploaded_features = extract_features(img_array)
    
    # ✅ Find best match
    best_match, similarity = compare_fingerprints(uploaded_features)
    
    # Generate person information if there's a match
    person_info = generate_person_info() if best_match else None
    
    # Format result
    match_result = {
        'best_match': best_match,
        'similarity': f"{similarity:.2%}",
        'person_info': person_info
    }

    return render_template('result.html', 
                         result=match_result, 
                         uploaded_image=filename,
                         matched_image=best_match)

if __name__ == '__main__':
    app.run(debug=True)
