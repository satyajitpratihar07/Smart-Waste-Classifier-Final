"""
Smart Waste Classifier - Streamlit Web Application
Professional dark-themed UI for waste classification
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os
from pathlib import Path
import yaml
import logging

# Configure page
st.set_page_config(
    page_title="♻️ Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22c55e, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11162a 0%, #0b1020 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(34, 197, 94, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding: 0 15px;
        margin: 10px 0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .tip-box {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .stProgress > div > div > div {
        background-color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_indices' not in st.session_state:
    st.session_state.class_indices = {}
if 'idx_to_class' not in st.session_state:
    st.session_state.idx_to_class = {}


@st.cache_resource
def load_model_and_config():
    """Load model and configuration with caching"""
    try:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model_path = os.path.join(
            config['paths']['model_dir'],
            config['paths']['model_name']
        )
        
        # Check if model exists
        if not os.path.exists(model_path):
            return None, None, "Model not found. Please train the model first."
        
        # Load model
        model = keras.models.load_model(model_path)
        
        # Load class indices
        class_indices_path = config['paths']['class_indices']
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            idx_to_class = {v: k for k, v in class_indices.items()}
        else:
            # Default class indices if file doesn't exist
            class_indices = {'Recyclable': 0, 'Organic': 1, 'Non-Recyclable': 2}
            idx_to_class = {0: 'Recyclable', 1: 'Organic', 2: 'Non-Recyclable'}
        
        return model, class_indices, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image
        target_size: Target image size
        
    Returns:
        Preprocessed image array
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_sustainability_tip(class_name: str) -> tuple:
    """
    Get sustainability tip and color for a waste class
    
    Args:
        class_name: Name of the waste class
        
    Returns:
        Tuple of (tip, color, emoji)
    """
    tips = {
        'Recyclable': (
            "♻️ Rinse containers before recycling to prevent contamination.",
            "#22c55e",
            "♻️"
        ),
        'Organic': (
            "🌿 Compost organic waste to enrich soil and reduce landfill methane.",
            "#f59e0b",
            "🌿"
        ),
        'Non-Recyclable': (
            "🚯 Avoid single-use plastics whenever possible. Consider reusable alternatives.",
            "#ef4444",
            "🚯"
        )
    }
    return tips.get(class_name, ("Reduce, reuse, and recycle!", "#6b7280", "♻️"))


def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">♻️ Smart Waste Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #9ca3af;">AI-Powered Waste Classification System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Information")
        
        # Load model
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                model, class_indices, error = load_model_and_config()
                if error:
                    st.error(error)
                    st.stop()
                else:
                    st.session_state.model = model
                    st.session_state.class_indices = class_indices
                    st.session_state.idx_to_class = {v: k for k, v in class_indices.items()}
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
        
        # Model info
        st.subheader("📊 Model Details")
        st.info(f"**Classes:** {len(st.session_state.class_indices)}")
        st.info(f"**Classes:** {', '.join(st.session_state.class_indices.keys())}")
        
        st.subheader("💡 Tips")
        st.markdown("""
        - Upload clear, well-lit images
        - Ensure waste item is centered
        - Works best with single items
        - Supports JPG, PNG formats
        """)
        
        st.subheader("📁 Dataset Info")
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            dataset_path = config['dataset']['path']
            if os.path.exists(dataset_path):
                st.success(f"Dataset found at: `{dataset_path}`")
            else:
                st.warning(f"Dataset path: `{dataset_path}`")
        except:
            pass
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of waste to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Classify Waste", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    img_array = preprocess_image(image)
                    
                    # Make prediction
                    predictions = st.session_state.model.predict(img_array, verbose=0)[0]
                    predicted_idx = np.argmax(predictions)
                    predicted_class = st.session_state.idx_to_class[predicted_idx]
                    confidence = predictions[predicted_idx]
                    
                    # Store in session state (convert numpy types to Python types)
                    st.session_state.prediction = {
                        'class': predicted_class,
                        'confidence': float(confidence),
                        'all_predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                    }
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            class_name = pred['class']
            confidence = pred['confidence']
            all_preds = pred['all_predictions']
            
            # Get tip and color
            tip, color, emoji = get_sustainability_tip(class_name)
            
            # Prediction card
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color: {color}; margin-bottom: 1rem;">
                    {emoji} {class_name}
                </h2>
                <h3 style="color: #e6e9ef; font-size: 2rem; margin: 1rem 0;">
                    {confidence*100:.2f}% Confidence
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(confidence)
            
            # All class probabilities
            st.subheader("📈 Class Probabilities")
            class_names = sorted(st.session_state.class_indices.keys())
            colors = {
                'Recyclable': '#22c55e',
                'Organic': '#f59e0b',
                'Non-Recyclable': '#ef4444'
            }
            
            for i, cls in enumerate(class_names):
                prob = all_preds[i]
                cls_color = colors.get(cls, '#6b7280')
                
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: 600;">{cls}</span>
                        <span style="color: {cls_color};">{prob*100:.2f}%</span>
                    </div>
                    <div style="background: rgba(107, 114, 128, 0.2); border-radius: 8px; height: 25px; overflow: hidden;">
                        <div style="background: {cls_color}; height: 100%; width: {prob*100}%; transition: width 0.3s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sustainability tip
            st.markdown(f"""
            <div class="tip-box">
                <h4 style="color: {color}; margin-bottom: 0.5rem;">💡 Sustainability Tip</h4>
                <p style="color: #e6e9ef; margin: 0;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload an image and click 'Classify Waste' to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #6b7280;'>Built with ❤️ using TensorFlow & Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()

