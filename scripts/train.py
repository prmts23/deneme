#!/usr/bin/env python3
"""
Script to train ABCD pattern detection model
"""
import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abcd_pattern import (
    Trainer,
    ModelConfig,
    DatasetConfig,
    setup_logging
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ABCD pattern detection model'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data_abcd',
        help='Dataset directory'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/abcd_detector.h5',
        help='Path to save trained model'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='TensorBoard log directory'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )

    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience'
    )

    parser.add_argument(
        '--transfer-learning',
        action='store_true',
        help='Use transfer learning (EfficientNetB0)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Configure model
    model_config = ModelConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping
    )

    # Configure dataset
    dataset_config = DatasetConfig(
        validation_split=args.validation_split
    )

    # Create trainer
    trainer = Trainer(
        model_config=model_config,
        dataset_config=dataset_config
    )

    print("="*60)
    print("ABCD Pattern Model Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Model output: {args.model_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Validation split: {args.validation_split}")
    print(f"  Early stopping: {args.early_stopping}")
    print(f"  Transfer learning: {args.transfer_learning}")
    print("\n" + "="*60)

    # Load datasets
    print("\nLoading datasets...")
    train_ds, val_ds = trainer.load_dataset(
        args.data_dir,
        validation_split=args.validation_split
    )

    # Train model
    print("\nStarting training...")
    print("="*60 + "\n")

    start_time = datetime.now()

    history = trainer.train(
        train_ds,
        val_ds,
        model_save_path=args.model_path,
        log_dir=args.log_dir,
        use_transfer_learning=args.transfer_learning
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"\nTraining time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"Model saved to: {args.model_path}")

    # Print final metrics
    final_epoch = len(history.history['loss'])
    print(f"\nFinal epoch: {final_epoch}/{args.epochs}")

    if 'loss' in history.history:
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"\nFinal metrics:")
        print(f"  Train loss: {final_train_loss:.4f}")
        print(f"  Val loss: {final_val_loss:.4f}")

    if 'accuracy' in history.history:
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"  Train accuracy: {final_train_acc:.4f}")
        print(f"  Val accuracy: {final_val_acc:.4f}")

    if 'precision' in history.history:
        final_train_prec = history.history['precision'][-1]
        final_val_prec = history.history['val_precision'][-1]
        print(f"  Train precision: {final_train_prec:.4f}")
        print(f"  Val precision: {final_val_prec:.4f}")

    if 'recall' in history.history:
        final_train_rec = history.history['recall'][-1]
        final_val_rec = history.history['val_recall'][-1]
        print(f"  Train recall: {final_train_rec:.4f}")
        print(f"  Val recall: {final_val_rec:.4f}")

    # Save training history
    history_path = args.model_path.replace('.h5', '_history.json')
    trainer.save_training_history(history_path)
    print(f"\nTraining history saved to: {history_path}")

    # Plot training history
    plot_path = args.model_path.replace('.h5', '_training_plot.png')
    trainer.plot_training_history(plot_path)
    print(f"Training plot saved to: {plot_path}")

    print("\n✓ Training complete!")
    print(f"\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
