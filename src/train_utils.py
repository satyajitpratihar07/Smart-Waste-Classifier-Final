"""
Training Utilities Module
Callbacks, plotting, and training helpers
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_training_callbacks(config: dict, outputs_dir: str) -> list:
    """
    Create list of training callbacks
    
    Args:
        config: Configuration dictionary
        outputs_dir: Directory to save outputs
        
    Returns:
        List of callback instances
    """
    callbacks = []
    training_config = config['training']
    
    # Early stopping
    early_stop_config = training_config['early_stopping']
    early_stopping = EarlyStopping(
        monitor=early_stop_config['monitor'],
        patience=early_stop_config['patience'],
        restore_best_weights=early_stop_config['restore_best_weights'],
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr_config = training_config['reduce_lr']
    reduce_lr = ReduceLROnPlateau(
        monitor=reduce_lr_config['monitor'],
        factor=reduce_lr_config['factor'],
        patience=reduce_lr_config['patience'],
        min_lr=reduce_lr_config['min_lr'],
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    model_dir = config['paths']['model_dir']
    model_name = config['paths']['model_name']
    checkpoint_path = os.path.join(model_dir, 'best_model.h5')
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
    callbacks.append(checkpoint)
    
    # CSV logger
    os.makedirs(outputs_dir, exist_ok=True)
    csv_logger = CSVLogger(
        filename=os.path.join(outputs_dir, 'training_log.csv'),
        append=False
    )
    callbacks.append(csv_logger)
    
    logger.info(f"Created {len(callbacks)} training callbacks")
    return callbacks


def plot_training_history(history: Dict, outputs_dir: str):
    """
    Plot and save training history (accuracy and loss)
    
    Args:
        history: Training history dictionary from model.fit()
        outputs_dir: Directory to save plots
    """
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training history plots to {outputs_dir}")
    
    # Individual plots
    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2.5, color='#22c55e')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5, color='#3b82f6')
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2.5, color='#ef4444')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='#f59e0b')
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_model(model: tf.keras.Model, config: dict):
    """
    Save trained model to disk
    
    Args:
        model: Trained Keras model
        config: Configuration dictionary
    """
    model_dir = config['paths']['model_dir']
    model_name = config['paths']['model_name']
    model_path = os.path.join(model_dir, model_name)
    
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")


def log_device_info():
    """Log information about available devices (GPU/CPU)"""
    logger.info("=" * 50)
    logger.info("Device Information")
    logger.info("=" * 50)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU Available: Yes ({len(gpus)} device(s))")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"    Memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"    Could not set memory growth: {e}")
    else:
        logger.info("GPU Available: No (using CPU)")
    
    logger.info(f"TensorFlow Version: {tf.__version__}")
    logger.info("=" * 50)

