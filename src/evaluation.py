"""
Evaluation Module
Generates confusion matrix, ROC curves, and classification reports
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from typing import Tuple, Dict
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: list, outputs_dir: str):
    """
    Generate and save confusion matrix
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        class_names: List of class names
        outputs_dir: Directory to save the plot
    """
    # Convert one-hot to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {outputs_dir}")


def generate_roc_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                       class_names: list, outputs_dir: str):
    """
    Generate and save ROC curves for each class
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        outputs_dir: Directory to save the plot
    """
    n_classes = len(class_names)
    colors = ['#22c55e', '#f59e0b', '#ef4444']  # Green, Orange, Red
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        # Get binary labels and probabilities for this class
        y_true_binary = y_true[:, i]
        y_pred_binary = y_pred_proba[:, i]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(outputs_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curves to {outputs_dir}")


def generate_pr_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      class_names: list, outputs_dir: str):
    """
    Generate and save Precision-Recall curves for each class
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        outputs_dir: Directory to save the plot
    """
    n_classes = len(class_names)
    colors = ['#22c55e', '#f59e0b', '#ef4444']  # Green, Orange, Red
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        # Get binary labels and probabilities for this class
        y_true_binary = y_true[:, i]
        y_pred_binary = y_pred_proba[:, i]
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        avg_precision = average_precision_score(y_true_binary, y_pred_binary)
        
        plt.plot(
            recall, precision,
            color=colors[i % len(colors)],
            lw=2,
            label=f'{class_names[i]} (AP = {avg_precision:.3f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(outputs_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PR curves to {outputs_dir}")


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: list, outputs_dir: str) -> str:
    """
    Generate and save classification report
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        class_names: List of class names
        outputs_dir: Directory to save the report
        
    Returns:
        Classification report as string
    """
    # Convert one-hot to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate report
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names,
        digits=4
    )
    
    # Save to file
    os.makedirs(outputs_dir, exist_ok=True)
    report_path = os.path.join(outputs_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    
    logger.info(f"Saved classification report to {report_path}")
    return report


def evaluate_model(model, val_generator, class_names: list, outputs_dir: str):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained Keras model
        val_generator: Validation data generator
        class_names: List of class names
        outputs_dir: Directory to save evaluation outputs
    """
    logger.info("Starting model evaluation...")
    
    # Get predictions
    logger.info("Generating predictions...")
    y_true = []
    y_pred_proba = []
    
    # Reset generator
    val_generator.reset()
    
    # Collect predictions in batches
    for i in range(len(val_generator)):
        batch_x, batch_y = val_generator[i]
        y_true.append(batch_y)
        y_pred_proba.append(model.predict(batch_x, verbose=0))
    
    # Concatenate all batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_onehot = np.eye(len(class_names))[y_pred]
    
    # Generate all evaluation metrics
    logger.info("Generating evaluation plots...")
    generate_confusion_matrix(y_true, y_pred_onehot, class_names, outputs_dir)
    generate_roc_curves(y_true, y_pred_proba, class_names, outputs_dir)
    generate_pr_curves(y_true, y_pred_proba, class_names, outputs_dir)
    report = generate_classification_report(y_true, y_pred_onehot, class_names, outputs_dir)
    
    logger.info("Evaluation complete!")
    return report

