"""
About Page
Project information and documentation
"""

import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About Smart Waste Classifier")

# Project Overview
st.header("🎯 Project Overview")

st.markdown("""
**Smart Waste Classifier** is an AI-powered deep learning system designed to classify waste images into three categories:

- **♻️ Recyclable** - Items that can be recycled (plastic bottles, paper, metal, etc.)
- **🌿 Organic** - Biodegradable waste (food scraps, garden waste, etc.)
- **🚯 Non-Recyclable** - Items that cannot be recycled (certain plastics, contaminated materials, etc.)

This application uses **transfer learning** with MobileNetV2, a lightweight and efficient convolutional neural network pre-trained on ImageNet, to achieve high accuracy in waste classification.
""")

# Model Architecture
st.header("🧠 Model Architecture")

st.markdown("""
### Transfer Learning Approach

The model uses **MobileNetV2** as the base architecture, which provides:

- ✅ **Efficiency**: Lightweight model suitable for deployment
- ✅ **Accuracy**: Pre-trained on ImageNet for robust feature extraction
- ✅ **Speed**: Fast inference times for real-time classification

### Architecture Details

1. **Base Model**: MobileNetV2 (ImageNet weights, frozen initially)
2. **Feature Extraction**: Global Average Pooling
3. **Regularization**: Batch Normalization + Dropout
4. **Classification Head**: Dense layers with Softmax activation

### Training Strategy

- **Phase 1**: Train classification head with frozen base model
- **Phase 2** (Optional): Fine-tune top layers for improved accuracy
""")

# Dataset
st.header("📊 Dataset")

st.markdown("""
The model is trained on a dataset organized in the following structure:

```
dataset/
├── Recyclable/
│   └── [images]
├── Organic/
│   └── [images]
└── Non-Recyclable/
    └── [images]
```

### Data Augmentation

To improve model generalization, the following augmentations are applied during training:

- Rotation (±20°)
- Translation (width/height shift)
- Shear transformation
- Zoom (up to 20%)
- Horizontal flip
- Brightness adjustment

### Train/Validation Split

- **Training**: 80% of the dataset
- **Validation**: 20% of the dataset
""")

# Technologies Used
st.header("🛠️ Technologies Used")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Machine Learning")
    st.markdown("""
    - **TensorFlow 2.x** - Deep learning framework
    - **Keras** - High-level neural network API
    - **MobileNetV2** - Pre-trained CNN architecture
    - **scikit-learn** - Evaluation metrics
    """)

with col2:
    st.subheader("Web Application")
    st.markdown("""
    - **Streamlit** - Web app framework
    - **Pillow (PIL)** - Image processing
    - **NumPy** - Numerical computations
    - **Matplotlib** - Visualization
    """)

# Features
st.header("✨ Key Features")

st.markdown("""
- 🎯 **High Accuracy**: Transfer learning with MobileNetV2
- 🚀 **Fast Inference**: Optimized for real-time predictions
- 🎨 **Beautiful UI**: Modern dark-themed interface
- 📊 **Comprehensive Metrics**: Training history, confusion matrix, ROC/PR curves
- ⚙️ **Configurable**: YAML-based configuration for easy customization
- 📱 **Responsive**: Works on desktop and mobile devices
- 🔄 **Batch Processing**: Support for multiple image predictions
- 💡 **Sustainability Tips**: Educational tips based on classification results
""")

# Usage
st.header("📖 How to Use")

st.markdown("""
### Training the Model

1. Organize your dataset in the specified folder structure
2. Configure hyperparameters in `config.yaml`
3. Run training:
   ```bash
   python train.py
   ```

### Using the Web App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload an image of waste
3. Click "Classify Waste" to get predictions
4. View confidence scores and sustainability tips

### Deployment

The app can be deployed to:
- **Streamlit Cloud**: Push to GitHub and deploy
- **Render**: Use the provided Procfile
- **Docker**: Build and run the Docker container
""")

# Performance
st.header("📈 Performance Metrics")

st.info("""
The model performance depends on:
- Dataset size and quality
- Training hyperparameters
- Image quality and preprocessing
- Model architecture choices

Typical performance metrics:
- **Accuracy**: 85-95% (varies by dataset)
- **Inference Time**: <100ms per image (GPU)
- **Model Size**: ~15-20 MB
""")

# Contributing
st.header("🤝 Contributing")

st.markdown("""
This is a professional-grade project designed for:
- Educational purposes
- Research and development
- Production deployment
- Customization and extension

Feel free to:
- Modify the architecture
- Add new classes
- Improve the UI
- Optimize performance
""")

# License & Credits
st.header("📄 License & Credits")

st.markdown("""
### Built With

- TensorFlow/Keras for deep learning
- Streamlit for web interface
- MobileNetV2 architecture (Google)
- Various open-source Python libraries

### Acknowledgments

- ImageNet dataset for pre-trained weights
- TensorFlow team for the framework
- Streamlit team for the web framework
""")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6b7280;'>Built with ❤️ for a sustainable future</p>",
    unsafe_allow_html=True
)

