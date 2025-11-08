# 🧠 Smart Waste Classifier — Pro Edition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI-Powered Deep Learning System for Waste Classification**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#-deployment) • [Architecture](#-architecture)

</div>

---

## 🎯 Overview

**Smart Waste Classifier** is a production-ready deep learning application that classifies waste images into three categories:

- **♻️ Recyclable** - Items that can be recycled (plastic bottles, paper, metal, etc.)
- **🌿 Organic** - Biodegradable waste (food scraps, garden waste, etc.)
- **🚯 Non-Recyclable** - Items that cannot be recycled (certain plastics, contaminated materials, etc.)

The system uses **transfer learning** with MobileNetV2, providing high accuracy while maintaining efficiency for real-time deployment.

---

## ✨ Features

### 🧠 Machine Learning
- ☑️ **Transfer Learning**: MobileNetV2 pre-trained on ImageNet
- ☑️ **Data Augmentation**: Rotation, zoom, brightness, flip, shear
- ☑️ **Fine-tuning Support**: Optional fine-tuning of top layers
- ☑️ **Comprehensive Evaluation**: Confusion matrix, ROC/PR curves, classification reports
- ☑️ **Early Stopping & LR Scheduling**: Automatic training optimization

### 💻 Web Application
- ✅ **Modern Dark UI**: Professional Tailwind-inspired theme
- ✅ **Real-time Predictions**: Fast inference with confidence scores
- ✅ **Batch Processing**: Support for multiple image predictions
- ✅ **Sustainability Tips**: Educational tips based on classification
- ✅ **Interactive Metrics**: Training history, evaluation plots
- ✅ **Responsive Design**: Works on desktop and mobile

### 🛠️ Development
- ☑️ **Modular Architecture**: Clean, maintainable code structure
- ☑️ **YAML Configuration**: Easy hyperparameter tuning
- ☑️ **GPU/CPU Support**: Automatic device detection
- ☑️ **Docker Support**: Containerized deployment
- ☑️** **Comprehensive Logging**: Detailed training logs

---

## 📁 Project Structure

```
SmartWasteClassifier_Pro/
├── app.py                     # Streamlit web app (main UI)
├── train.py                   # Training pipeline
├── config.yaml                # Configurable hyperparameters
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment config (Render/Heroku)
├── Dockerfile                 # Docker container configuration
├── README.md                  # This file
│
├── .streamlit/
│   └── config.toml            # Streamlit theme customization
│
├── src/                       # Modular ML backend
│   ├── data.py                # Data loading & augmentation
│   ├── modeling.py            # MobileNetV2 architecture
│   ├── train_utils.py         # Callbacks, plots, utilities
│   └── evaluation.py          # Confusion matrix, ROC/PR curves
│
├── pages/                     # Streamlit multi-page app
│   ├── 1_📈_Metrics.py        # View training metrics
│   ├── 2_🧠_Model_Info.py     # Model summary page
│   └── 3_ℹ️_About.py          # About & documentation
│
├── dataset/                   # Images folder (each subfolder = class)
│   ├── Recyclable/
│   ├── Organic/
│   └── Non-Recyclable/
│
├── model/                     # Trained models
│   ├── waste_classifier.h5
│   └── class_indices.json
│
└── outputs/                   # Training logs & plots
    ├── training_accuracy.png
    ├── training_loss.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── pr_curves.png
    ├── classification_report.txt
    └── training_log.csv
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd SmartWasteClassifier_Pro
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset

Organize your dataset in the following structure:

```
dataset/
├── Recyclable/
│   └── [recyclable images]
├── Organic/
│   └── [organic images]
└── Non-Recyclable/
    └── [non-recyclable images]
```

**Note:** The data loader also supports nested structures (e.g., `dataset/organic/organic/`).

---

## 📖 Usage

### Training the Model

1. **Configure Hyperparameters** (optional):
   Edit `config.yaml` to adjust:
   - Learning rate
   - Batch size
   - Epochs
   - Data augmentation parameters
   - Fine-tuning settings

2. **Start Training**:
   ```bash
   python train.py
   ```

   Or with custom config:
   ```bash
   python train.py --config custom_config.yaml
   ```

3. **Monitor Training**:
   - Training progress is logged to console
   - Metrics are saved to `outputs/` directory
   - Best model is saved to `model/waste_classifier.h5`

### Running the Web Application

1. **Start Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Access the App**:
   - Open your browser to `http://localhost:8501`
   - Upload an image and click "Classify Waste"

3. **Navigate Pages**:
   - **Home**: Main classification interface
   - **📈 Metrics**: View training history and evaluation plots
   - **🧠 Model Info**: See model architecture and configuration
   - **ℹ️ About**: Project documentation

---

## 🏗️ Architecture

### Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 Base (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(3, Softmax) → [Recyclable, Organic, Non-Recyclable]
```

### Training Pipeline

1. **Data Loading**: Load images from class folders, apply augmentations
2. **Base Training**: Train classification head with frozen MobileNetV2
3. **Fine-tuning** (optional): Unfreeze top layers and continue training
4. **Evaluation**: Generate metrics, plots, and reports
5. **Model Saving**: Save best model and class indices

### Key Components

- **`src/data.py`**: Handles dataset loading, augmentation, and preprocessing
- **`src/modeling.py`**: Builds MobileNetV2-based classifier with transfer learning
- **`src/train_utils.py`**: Training callbacks, history plotting, model saving
- **`src/evaluation.py`**: Comprehensive evaluation metrics and visualizations

---

## 🚢 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy!

### Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the provided `Procfile`
4. Set build command: `pip install -r requirements.txt`
5. Deploy!

### Docker

1. **Build the image**:
   ```bash
   docker build -t waste-classifier .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 waste-classifier
   ```

3. **Access the app**:
   Open `http://localhost:8501` in your browser

### Local Production

For production deployment, use a process manager like `gunicorn` or `supervisor`:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Dataset Settings
```yaml
dataset:
  path: "dataset"
  image_size: [224, 224]
  batch_size: 32
  validation_split: 0.2
```

### Model Settings
```yaml
model:
  base_model: "MobileNetV2"
  dropout_rate: 0.3
  dense_units: 256
  fine_tune_enabled: false
  fine_tune_layers: 30
```

### Training Settings
```yaml
training:
  epochs: 50
  learning_rate: 0.001
  early_stopping:
    patience: 10
```

---

## 📊 Performance

Typical performance metrics (varies by dataset):

- **Accuracy**: 85-95%
- **Inference Time**: <100ms per image (GPU), <500ms (CPU)
- **Model Size**: ~15-20 MB
- **Training Time**: ~30-60 minutes (GPU, 10k images)

---

## 🧪 Example Output

### Classification Result
```
♻️ Recyclable
Confidence: 94.23%

Class Probabilities:
- Recyclable: 94.23% ████████████████████
- Organic: 4.12% █
- Non-Recyclable: 1.65% █

💡 Sustainability Tip:
Rinse containers before recycling to prevent contamination.
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found error**:
   - Train the model first: `python train.py`
   - Check `model/waste_classifier.h5` exists

2. **Out of memory**:
   - Reduce `batch_size` in `config.yaml`
   - Use smaller `image_size`

3. **Slow training**:
   - Enable GPU if available
   - Reduce `epochs` or dataset size for testing

4. **Import errors**:
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** team for the deep learning framework
- **Streamlit** team for the web framework
- **Google** for MobileNetV2 architecture
- **ImageNet** dataset for pre-trained weights
