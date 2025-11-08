"""
Model Architecture Module
Builds MobileNetV2-based waste classifier with transfer learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteClassifierModel:
    """Builds and configures the waste classification model"""
    
    def __init__(self, config: dict, num_classes: int):
        """
        Initialize model builder
        
        Args:
            config: Configuration dictionary
            num_classes: Number of output classes
        """
        self.config = config
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        
    def build_base_model(self) -> Model:
        """
        Build MobileNetV2 base model with ImageNet weights
        
        Returns:
            Base model without top layers
        """
        model_config = self.config['model']
        input_shape = tuple(model_config['input_shape'])
        
        # Load MobileNetV2 pre-trained on ImageNet
        self.base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            alpha=1.0
        )
        
        # Freeze base model layers initially
        self.base_model.trainable = False
        
        logger.info(f"Loaded MobileNetV2 base model with {len(self.base_model.layers)} layers")
        return self.base_model
    
    def build_classifier_head(self, base_model: Model) -> Model:
        """
        Add classification head to base model
        
        Args:
            base_model: Pre-trained base model
            
        Returns:
            Complete model with classification head
        """
        model_config = self.config['model']
        
        # Add custom classification head
        inputs = keras.Input(shape=model_config['input_shape'])
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        
        # Batch normalization
        x = BatchNormalization()(x)
        
        # Dense layer with dropout
        x = Dense(
            model_config['dense_units'],
            activation='relu',
            name='dense_layer'
        )(x)
        x = Dropout(model_config['dropout_rate'])(x)
        
        # Output layer
        outputs = Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        self.model = Model(inputs, outputs, name='waste_classifier')
        
        logger.info("Built classification head")
        return self.model
    
    def compile_model(self, learning_rate: float = None) -> Model:
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer (uses config if None)
            
        Returns:
            Compiled model
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate']
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Compiled model with learning rate: {learning_rate}")
        return self.model
    
    def enable_fine_tuning(self):
        """
        Unfreeze top layers for fine-tuning
        """
        if self.base_model is None:
            raise ValueError("Base model must be built before fine-tuning")
        
        fine_tune_layers = self.config['model']['fine_tune_layers']
        
        # Unfreeze top N layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        fine_tune_lr = self.config['training']['fine_tune_learning_rate']
        self.compile_model(learning_rate=fine_tune_lr)
        
        trainable_count = sum([1 for layer in self.base_model.layers if layer.trainable])
        logger.info(f"Enabled fine-tuning for {trainable_count} layers")
        logger.info(f"Fine-tuning learning rate: {fine_tune_lr}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary as string
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def build_complete_model(self) -> Model:
        """
        Build complete model (base + head) and compile
        
        Returns:
            Compiled model ready for training
        """
        base_model = self.build_base_model()
        model = self.build_classifier_head(base_model)
        model = self.compile_model()
        
        return model

