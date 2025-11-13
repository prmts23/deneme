"""
CNN Model for ABCD Pattern Detection
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Optional, Tuple
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ABCDPatternCNN:
    """CNN model for ABCD pattern detection"""

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize model

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model: Optional[keras.Model] = None

    def build_model(self) -> keras.Model:
        """
        Build CNN model architecture

        Returns:
            Compiled Keras model
        """
        logger.info("Building CNN model...")

        inputs = layers.Input(shape=self.config.input_shape)

        # Preprocessing
        x = layers.Rescaling(1./255)(inputs)

        # Data augmentation (optional, can be enabled during training)
        # x = layers.RandomFlip("horizontal")(x)
        # x = layers.RandomRotation(0.05)(x)
        # x = layers.RandomZoom(0.1)(x)

        # Convolutional blocks
        for i, filters in enumerate(self.config.conv_filters):
            x = layers.Conv2D(
                filters,
                self.config.conv_kernel_size,
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                name=f'conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.Conv2D(
                filters,
                self.config.conv_kernel_size,
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                name=f'conv_{i}_2'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}_2')(x)
            x = layers.MaxPooling2D(
                self.config.pool_size,
                name=f'pool_{i}'
            )(x)
            x = layers.Dropout(0.25, name=f'dropout_conv_{i}')(x)

        # Global pooling
        x = layers.GlobalAveragePooling2D(name='gap')(x)

        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_initializer='he_normal',
                name=f'dense_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i}')(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'dropout_dense_{i}')(x)

        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='abcd_pattern_detector')

        # Compile model
        optimizer = self._get_optimizer()
        metrics = self._get_metrics()

        model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=metrics
        )

        logger.info(f"Model built successfully with {model.count_params():,} parameters")

        self.model = model
        return model

    def build_transfer_learning_model(
        self,
        base_model_name: str = "EfficientNetB0",
        trainable_layers: int = 20
    ) -> keras.Model:
        """
        Build model using transfer learning

        Args:
            base_model_name: Name of base model (EfficientNetB0, ResNet50, etc.)
            trainable_layers: Number of top layers to make trainable

        Returns:
            Compiled Keras model
        """
        logger.info(f"Building transfer learning model with {base_model_name}...")

        # Get base model
        base_model_class = getattr(keras.applications, base_model_name)
        base_model = base_model_class(
            include_top=False,
            weights='imagenet',
            input_shape=self.config.input_shape
        )

        # Freeze base model layers
        base_model.trainable = False

        # Build model
        inputs = layers.Input(shape=self.config.input_shape)
        x = layers.Rescaling(1./255)(inputs)

        # Preprocess input for base model
        preprocess_fn = getattr(
            keras.applications,
            f"{base_model_name.lower()}"
        ).preprocess_input
        x = preprocess_fn(x)

        # Base model
        x = base_model(x, training=False)

        # Top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        for units in self.config.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile
        optimizer = self._get_optimizer()
        metrics = self._get_metrics()

        model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=metrics
        )

        logger.info(f"Transfer learning model built with {model.count_params():,} parameters")

        self.model = model
        return model

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get optimizer from config"""
        if self.config.optimizer.lower() == "adam":
            return keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            return keras.optimizers.SGD(
                learning_rate=self.config.learning_rate,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == "adamw":
            return keras.optimizers.AdamW(learning_rate=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _get_metrics(self) -> list:
        """Get metrics from config"""
        metric_map = {
            "accuracy": keras.metrics.BinaryAccuracy(name='accuracy'),
            "precision": keras.metrics.Precision(name='precision'),
            "recall": keras.metrics.Recall(name='recall'),
            "auc": keras.metrics.AUC(name='auc')
        }

        metrics = []
        for metric_name in self.config.metrics:
            if metric_name.lower() in metric_map:
                metrics.append(metric_map[metric_name.lower()])
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return metrics

    def summary(self) -> None:
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.summary()

    def save(self, filepath: str) -> None:
        """
        Save model to file

        Args:
            filepath: Output file path
        """
        if self.model is None:
            raise ValueError("Model not built yet")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> keras.Model:
        """
        Load model from file

        Args:
            filepath: Model file path

        Returns:
            Loaded model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

    def predict(self, x, threshold: float = 0.5) -> Tuple[float, bool]:
        """
        Make prediction

        Args:
            x: Input image or batch
            threshold: Classification threshold

        Returns:
            (confidence, is_pattern)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        pred = self.model.predict(x, verbose=0)
        confidence = float(pred[0][0])
        is_pattern = confidence >= threshold

        return confidence, is_pattern
