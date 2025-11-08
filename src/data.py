"""
Data Loading and Augmentation Module
Handles dataset loading, preprocessing, and augmentation
"""

import os
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteDataLoader:
    """Handles loading and preprocessing of waste classification dataset"""
    
    def __init__(self, config: dict):
        """
        Initialize data loader with configuration
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.dataset_path = Path(config['dataset']['path'])
        self.image_size = tuple(config['dataset']['image_size'])
        self.batch_size = config['dataset']['batch_size']
        self.validation_split = config['dataset']['validation_split']
        self.seed = config['dataset']['seed']
        self.class_indices = {}
        self.num_classes = 0
        self.temp_data_dir = None  # For nested structure handling
        
    def discover_classes(self) -> Dict[str, int]:
        """
        Discover class names from dataset directory structure
        
        Returns:
            Dictionary mapping class names to indices
        """
        classes = []
        
        # Look for subdirectories in dataset path
        for item in self.dataset_path.iterdir():
            if item.is_dir():
                # Handle nested structure (e.g., dataset/organic/organic/)
                class_name = item.name
                # Check if there are subdirectories inside
                subdirs = [d for d in item.iterdir() if d.is_dir()]
                if subdirs:
                    # Use the nested directory name
                    class_name = subdirs[0].name
                
                # Normalize class names
                class_name = class_name.capitalize()
                if class_name not in classes:
                    classes.append(class_name)
        
        # Also check for direct class folders
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name.capitalize() not in classes:
                classes.append(item.name.capitalize())
        
        # Sort classes for consistency
        classes = sorted(classes)
        self.class_indices = {cls: idx for idx, cls in enumerate(classes)}
        self.num_classes = len(classes)
        
        logger.info(f"Discovered {self.num_classes} classes: {list(self.class_indices.keys())}")
        return self.class_indices
    
    def get_data_generators(self) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators with augmentation
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        aug_config = self.config['augmentation']
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=aug_config['rotation_range'],
            width_shift_range=aug_config['width_shift_range'],
            height_shift_range=aug_config['height_shift_range'],
            shear_range=aug_config['shear_range'],
            zoom_range=aug_config['zoom_range'],
            horizontal_flip=aug_config['horizontal_flip'],
            brightness_range=aug_config['brightness_range'],
            fill_mode=aug_config['fill_mode'],
            validation_split=self.validation_split
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        return train_datagen, val_datagen
    
    def _prepare_nested_structure(self) -> Path:
        """
        Handle nested directory structure by creating symlinks
        For structure like dataset/organic/organic/images, creates dataset_temp/organic -> dataset/organic/organic
        
        Returns:
            Path to the prepared data directory
        """
        data_dir = self.dataset_path
        has_nested = False
        nested_classes = {}
        
        # Check for nested structure
        for item in data_dir.iterdir():
            if item.is_dir():
                nested = item / item.name
                if nested.exists() and nested.is_dir():
                    # Check if nested folder has images
                    has_images = any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] 
                                    for f in nested.iterdir() if f.is_file())
                    if has_images:
                        has_nested = True
                        nested_classes[item.name] = nested
        
        if not has_nested:
            logger.info("No nested structure detected, using dataset as-is")
            return data_dir
        
        # Create temporary directory with flattened structure
        logger.info("Detected nested directory structure - creating temporary flattened structure")
        temp_dir = Path(tempfile.mkdtemp(prefix='waste_dataset_'))
        self.temp_data_dir = temp_dir
        
        # Create symlinks/junctions for each class
        for class_name, nested_path in nested_classes.items():
            class_dir = temp_dir / class_name
            
            # Try different methods to create links
            success = False
            
            # Method 1: Try symlink (works on Linux/Mac and Windows 10+ with developer mode)
            try:
                if class_dir.exists():
                    class_dir.rmdir()
                class_dir.symlink_to(nested_path, target_is_directory=True)
                logger.info(f"Created symlink: {class_dir} -> {nested_path}")
                success = True
            except (OSError, AttributeError) as e:
                # Method 2: Try Windows junction (requires admin or developer mode)
                if os.name == 'nt':
                    try:
                        import subprocess
                        if class_dir.exists():
                            class_dir.rmdir()
                        # Use mklink /J for directory junction
                        result = subprocess.run(
                            ['cmd', '/c', 'mklink', '/J', str(class_dir), str(nested_path)],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        logger.info(f"Created junction: {class_dir} -> {nested_path}")
                        success = True
                    except (subprocess.CalledProcessError, FileNotFoundError) as e2:
                        logger.debug(f"Junction creation failed: {e2}")
            
            # Method 3: Fallback - use hard links for files (no admin needed, but slower)
            if not success:
                logger.warning(f"Symlink/junction creation failed for {class_name}")
                logger.info(f"Using hard links for files (no admin required)...")
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create hard links for each image file
                image_files = [f for f in nested_path.iterdir() 
                             if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                linked_count = 0
                copied_count = 0
                
                for i, img_file in enumerate(image_files):
                    link_path = class_dir / img_file.name
                    if link_path.exists():
                        continue
                    
                    try:
                        # Try to create hard link
                        os.link(str(img_file), str(link_path))
                        linked_count += 1
                    except (OSError, AttributeError):
                        # If hard link fails (e.g., different drive), copy the file
                        shutil.copy2(img_file, link_path)
                        copied_count += 1
                    
                    # Log progress every 1000 files
                    if (i + 1) % 1000 == 0:
                        logger.info(f"  Processed {i + 1}/{len(image_files)} files for {class_name}...")
                
                logger.info(f"Created {linked_count} hard links and {copied_count} copies for {class_name} ({len(image_files)} total)")
        
        logger.info(f"Created temporary data directory: {temp_dir}")
        return temp_dir
    
    def get_data_flows(self) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """
        Create data flow generators for training and validation
        
        Returns:
            Tuple of (train_flow, val_flow)
        """
        train_datagen, val_datagen = self.get_data_generators()
        
        # Prepare data directory (handle nested structure if needed)
        data_dir = self._prepare_nested_structure()
        
        logger.info(f"Using data directory: {data_dir}")
        
        # Training flow
        train_flow = train_datagen.flow_from_directory(
            directory=str(data_dir),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            seed=self.seed,
            shuffle=True
        )
        
        # Validation flow
        val_flow = val_datagen.flow_from_directory(
            directory=str(data_dir),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            seed=self.seed,
            shuffle=False
        )
        
        # Update class indices from generator
        self.class_indices = train_flow.class_indices
        # Reverse to get class name from index
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        
        logger.info(f"Training samples: {train_flow.samples}")
        logger.info(f"Validation samples: {val_flow.samples}")
        logger.info(f"Class indices: {self.class_indices}")
        
        return train_flow, val_flow
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory if created"""
        if self.temp_data_dir and self.temp_data_dir.exists():
            try:
                shutil.rmtree(self.temp_data_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_data_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
    
    def save_class_indices(self, save_path: str):
        """
        Save class indices to JSON file
        
        Args:
            save_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.class_indices, f, indent=2)
        logger.info(f"Saved class indices to {save_path}")
    
    def load_class_indices(self, load_path: str) -> Dict[str, int]:
        """
        Load class indices from JSON file
        
        Args:
            load_path: Path to the JSON file
            
        Returns:
            Dictionary mapping class names to indices
        """
        with open(load_path, 'r') as f:
            self.class_indices = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        logger.info(f"Loaded class indices from {load_path}")
        return self.class_indices

