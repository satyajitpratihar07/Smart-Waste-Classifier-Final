"""
Training Metrics Page
Displays training history, confusion matrix, and evaluation metrics
"""

import streamlit as st
import os
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path

st.set_page_config(
    page_title="Training Metrics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Training Metrics & Evaluation")

# Load config
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    outputs_dir = config['paths']['outputs_dir']
except:
    outputs_dir = 'outputs'
    st.warning("Could not load config.yaml, using default paths")

# Check if outputs directory exists
if not os.path.exists(outputs_dir):
    st.error(f"❌ Outputs directory not found at: `{outputs_dir}`")
    st.info("💡 Please train the model first using: `python train.py`")
    st.stop()

# Training History Section
st.header("📊 Training History")

col1, col2 = st.columns(2)

with col1:
    accuracy_path = os.path.join(outputs_dir, 'training_accuracy.png')
    if os.path.exists(accuracy_path):
        st.subheader("Accuracy Curves")
        st.image(accuracy_path, use_container_width=True)
    else:
        st.info("Accuracy plot not available")

with col2:
    loss_path = os.path.join(outputs_dir, 'training_loss.png')
    if os.path.exists(loss_path):
        st.subheader("Loss Curves")
        st.image(loss_path, use_container_width=True)
    else:
        st.info("Loss plot not available")

# Training Log CSV
csv_path = os.path.join(outputs_dir, 'training_log.csv')
if os.path.exists(csv_path):
    st.header("📋 Training Log")
    df = pd.read_csv(csv_path)
    
    # Display metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Training Accuracy", f"{df['accuracy'].iloc[-1]:.4f}")
    with col2:
        st.metric("Final Validation Accuracy", f"{df['val_accuracy'].iloc[-1]:.4f}")
    with col3:
        st.metric("Final Training Loss", f"{df['loss'].iloc[-1]:.4f}")
    with col4:
        st.metric("Final Validation Loss", f"{df['val_loss'].iloc[-1]:.4f}")
    
    # Display full table
    with st.expander("📊 View Full Training History"):
        st.dataframe(df, use_container_width=True)

# Evaluation Metrics Section
st.header("🎯 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    cm_path = os.path.join(outputs_dir, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix")
        st.image(cm_path, use_container_width=True)
    else:
        st.info("Confusion matrix not available")

with col2:
    roc_path = os.path.join(outputs_dir, 'roc_curves.png')
    if os.path.exists(roc_path):
        st.subheader("ROC Curves")
        st.image(roc_path, use_container_width=True)
    else:
        st.info("ROC curves not available")

# PR Curves
pr_path = os.path.join(outputs_dir, 'pr_curves.png')
if os.path.exists(pr_path):
    st.subheader("Precision-Recall Curves")
    st.image(pr_path, use_container_width=True)

# Classification Report
report_path = os.path.join(outputs_dir, 'classification_report.txt')
if os.path.exists(report_path):
    st.header("📄 Classification Report")
    with open(report_path, 'r') as f:
        report = f.read()
    st.code(report, language='text')

# Additional Info
st.markdown("---")
st.info("💡 These metrics are generated during model training. Retrain the model to update metrics.")

