"""
Training Pipeline for ABCD Pattern Detector
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple, Dict
import logging
from datetime import datetime
import json

from .model import ABCDPatternCNN
from .config import ModelConfig, DatasetConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for ABCD pattern detector"""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        dataset_config: Optional[DatasetConfig] = None
    ):
        """
        Initialize trainer

        Args:
            model_config: Model configuration
            dataset_config: Dataset configuration
        """
        self.model_config = model_config or ModelConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.model_builder = ABCDPatternCNN(self.model_config)
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

    def load_dataset(
        self,
        data_dir: str,
        validation_split: Optional[float] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and validation datasets

        Args:
            data_dir: Dataset directory
            validation_split: Validation split ratio

        Returns:
            (train_ds, val_ds)
        """
        validation_split = validation_split or self.dataset_config.validation_split

        logger.info(f"Loading dataset from {data_dir}...")

        # Training dataset
        train_ds = keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="binary",
            image_size=self.dataset_config.image_size,
            batch_size=self.model_config.batch_size,
            validation_split=validation_split,
            subset="training",
            seed=42,
            shuffle=True
        )

        # Validation dataset
        val_ds = keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            label_mode="binary",
            image_size=self.dataset_config.image_size,
            batch_size=self.model_config.batch_size,
            validation_split=validation_split,
            subset="validation",
            seed=42,
            shuffle=True
        )

        # Optimize dataset performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        logger.info("Dataset loaded successfully")
        return train_ds, val_ds

    def get_callbacks(
        self,
        model_save_path: str,
        log_dir: Optional[str] = None
    ) -> list:
        """
        Get training callbacks

        Args:
            model_save_path: Path to save best model
            log_dir: TensorBoard log directory

        Returns:
            List of callbacks
        """
        callbacks = []

        # Model checkpoint
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stop_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.model_config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop_callback)

        # Reduce learning rate on plateau
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)

        # TensorBoard
        if log_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tb_log_dir = os.path.join(log_dir, timestamp)
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            )
            callbacks.append(tensorboard_callback)
            logger.info(f"TensorBoard logs: {tb_log_dir}")

        # CSV logger
        csv_path = model_save_path.replace('.h5', '_training.csv')
        csv_callback = keras.callbacks.CSVLogger(csv_path)
        callbacks.append(csv_callback)

        return callbacks

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        model_save_path: str,
        log_dir: Optional[str] = None,
        use_transfer_learning: bool = False
    ) -> keras.callbacks.History:
        """
        Train model

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            model_save_path: Path to save model
            log_dir: TensorBoard log directory
            use_transfer_learning: Use transfer learning

        Returns:
            Training history
        """
        logger.info("Starting training...")

        # Build model
        if use_transfer_learning:
            self.model = self.model_builder.build_transfer_learning_model()
        else:
            self.model = self.model_builder.build_model()

        # Print model summary
        self.model.summary()

        # Get callbacks
        callbacks = self.get_callbacks(model_save_path, log_dir)

        # Train
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.model_config.epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return self.history

    def evaluate(
        self,
        test_ds: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Evaluate model on test set

        Args:
            test_ds: Test dataset

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        logger.info("Evaluating model...")
        results = self.model.evaluate(test_ds, verbose=1)

        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(results[i])

        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def save_training_history(self, filepath: str) -> None:
        """
        Save training history to JSON

        Args:
            filepath: Output file path
        """
        if self.history is None:
            raise ValueError("No training history available")

        history_dict = {
            key: [float(val) for val in values]
            for key, values in self.history.history.items()
        }

        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Training history saved to {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history

        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        if self.history is None:
            raise ValueError("No training history available")

        import matplotlib.pyplot as plt

        history = self.history.history

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training History', fontsize=16)

        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Loss')
        axes[0, 0].grid(True)

        # Accuracy
        if 'accuracy' in history:
            axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
            axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].grid(True)

        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision')
            axes[1, 0].plot(history['val_precision'], label='Val Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].set_title('Precision')
            axes[1, 0].grid(True)

        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall')
            axes[1, 1].plot(history['val_recall'], label='Val Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].set_title('Recall')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
