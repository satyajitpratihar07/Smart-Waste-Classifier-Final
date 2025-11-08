"""
Model Information Page
Displays model architecture and configuration details
"""

import streamlit as st
import yaml
import os
import json
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Model Information",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Model Information")

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except:
    st.error("❌ Could not load config.yaml")
    st.stop()

# Model Architecture Section
st.header("🏗️ Model Architecture")

st.subheader("Base Model")
st.info(f"**Model:** {config['model']['base_model']}")
st.info(f"**Input Shape:** {config['model']['input_shape']}")
st.info(f"**Number of Classes:** {config['model']['num_classes']}")

st.subheader("Architecture Details")
architecture_info = f"""
**Transfer Learning:** MobileNetV2 (ImageNet weights)
**Classification Head:**
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense({config['model']['dense_units']}, ReLU)
  - Dropout({config['model']['dropout_rate']})
  - Dense({config['model']['num_classes']}, Softmax)

**Fine-tuning:** {'Enabled' if config['model']['fine_tune_enabled'] else 'Disabled'}
"""
st.markdown(architecture_info)

# Load and display model summary if available
model_path = os.path.join(
    config['paths']['model_dir'],
    config['paths']['model_name']
)

if os.path.exists(model_path):
    try:
        st.subheader("📋 Model Summary")
        with st.spinner("Loading model..."):
            model = keras.models.load_model(model_path)
            
            # Model summary
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            summary_text = '\n'.join(summary_list)
            
            with st.expander("View Full Model Summary"):
                st.code(summary_text, language='text')
            
            # Model statistics
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Parameters", f"{total_params:,}")
            with col2:
                st.metric("Trainable Parameters", f"{trainable_params:,}")
            with col3:
                st.metric("Non-Trainable Parameters", f"{non_trainable_params:,}")
    except Exception as e:
        st.warning(f"Could not load model: {str(e)}")
else:
    st.warning(f"Model not found at: `{model_path}`")

# Training Configuration
st.header("⚙️ Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hyperparameters")
    training_config = config['training']
    st.json({
        "Epochs": training_config['epochs'],
        "Initial Epochs": training_config['initial_epochs'],
        "Learning Rate": training_config['learning_rate'],
        "Fine-tune Learning Rate": training_config['fine_tune_learning_rate'],
        "Batch Size": config['dataset']['batch_size'],
        "Image Size": config['dataset']['image_size'],
        "Validation Split": config['dataset']['validation_split']
    })

with col2:
    st.subheader("Data Augmentation")
    aug_config = config['augmentation']
    st.json({
        "Rotation Range": aug_config['rotation_range'],
        "Width Shift": aug_config['width_shift_range'],
        "Height Shift": aug_config['height_shift_range'],
        "Shear Range": aug_config['shear_range'],
        "Zoom Range": aug_config['zoom_range'],
        "Horizontal Flip": aug_config['horizontal_flip'],
        "Brightness Range": aug_config['brightness_range']
    })

# Class Information
st.header("🏷️ Class Information")

class_indices_path = config['paths']['class_indices']
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    st.subheader("Class Labels")
    for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
        emoji_map = {
            'Recyclable': '♻️',
            'Organic': '🌿',
            'Non-Recyclable': '🚯'
        }
        emoji = emoji_map.get(class_name, '📦')
        st.info(f"{emoji} **{class_name}** (Index: {idx})")
else:
    st.warning(f"Class indices file not found at: `{class_indices_path}`")

# File Paths
st.header("📁 File Paths")
paths_config = config['paths']
st.json({
    "Model Directory": paths_config['model_dir'],
    "Model File": paths_config['model_name'],
    "Outputs Directory": paths_config['outputs_dir'],
    "Class Indices": paths_config['class_indices']
})

# System Information
st.header("💻 System Information")
st.info(f"**TensorFlow Version:** {tf.__version__}")
st.info(f"**Keras Version:** {keras.__version__}")

# GPU Information
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    st.success(f"✅ GPU Available: {len(gpus)} device(s)")
    for i, gpu in enumerate(gpus):
        st.info(f"  GPU {i}: {gpu.name}")
else:
    st.info("ℹ️ Using CPU (GPU not detected)")

