"""
Training Pipeline for Smart Waste Classifier
Main script to train the waste classification model
"""

import os
import yaml
import argparse
from pathlib import Path
import tensorflow as tf
from src.data import WasteDataLoader
from src.modeling import WasteClassifierModel
from src.train_utils import (
    get_training_callbacks,
    plot_training_history,
    save_model,
    log_device_info
)
from src.evaluation import evaluate_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Smart Waste Classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Log device information
    log_device_info()
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = WasteDataLoader(config)
    
    # Discover classes and get data flows
    data_loader.discover_classes()
    train_flow, val_flow = data_loader.get_data_flows()
    
    # Save class indices
    class_indices_path = config['paths']['class_indices']
    data_loader.save_class_indices(class_indices_path)
    
    # Get class names for evaluation
    class_names = sorted(data_loader.class_indices.keys())
    num_classes = len(class_names)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    
    # Build model
    logger.info("Building model...")
    model_builder = WasteClassifierModel(config, num_classes)
    model = model_builder.build_complete_model()
    
    # Display model summary
    logger.info("\n" + "=" * 50)
    logger.info("Model Architecture Summary")
    logger.info("=" * 50)
    model_builder.get_model_summary()
    
    # Get training callbacks
    outputs_dir = config['paths']['outputs_dir']
    callbacks = get_training_callbacks(config, outputs_dir)
    
    # Phase 1: Train with frozen base
    logger.info("\n" + "=" * 50)
    logger.info("Phase 1: Training with frozen base model")
    logger.info("=" * 50)
    
    training_config = config['training']
    initial_epochs = training_config['initial_epochs']
    
    history = model.fit(
        train_flow,
        epochs=initial_epochs,
        validation_data=val_flow,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning (if enabled)
    if config['model']['fine_tune_enabled']:
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Fine-tuning top layers")
        logger.info("=" * 50)
        
        # Enable fine-tuning
        model_builder.enable_fine_tuning()
        
        # Continue training
        fine_tune_epochs = training_config['epochs'] - initial_epochs
        fine_tune_history = model.fit(
            train_flow,
            initial_epoch=initial_epochs,
            epochs=training_config['epochs'],
            validation_data=val_flow,
            callbacks=callbacks,
            verbose=1
        )
        
        # Merge histories
        for key in history.history.keys():
            history.history[key].extend(fine_tune_history.history[key])
    
    # Plot training history
    logger.info("Generating training plots...")
    plot_training_history(history, outputs_dir)
    
    # Save final model
    logger.info("Saving final model...")
    save_model(model, config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluate_model(model, val_flow, class_names, outputs_dir)
    
    # Clean up temporary directory if created
    data_loader.cleanup_temp_dir()
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    logger.info(f"Model saved to: {os.path.join(config['paths']['model_dir'], config['paths']['model_name'])}")
    logger.info(f"Outputs saved to: {outputs_dir}")


if __name__ == '__main__':
    main()

