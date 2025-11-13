#!/usr/bin/env python3
"""
Example usage of ABCD Pattern Detector
This script demonstrates the complete workflow from data generation to backtesting
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import ABCD pattern modules
from abcd_pattern import (
    DatasetGenerator,
    Trainer,
    ABCDDetector,
    ABCDPatternGenerator,
    setup_logging,
    PatternConfig,
    DatasetConfig,
    ModelConfig,
    BacktestConfig
)
from backtest import ABCDBacktest


def example_1_generate_dataset():
    """Example 1: Generate training dataset"""
    print("\n" + "="*60)
    print("Example 1: Generate Training Dataset")
    print("="*60 + "\n")

    # Setup logging
    setup_logging(log_level="INFO")

    # Create generator with custom config
    pattern_config = PatternConfig(
        min_bars=10,
        max_bars=25,
        noise_level=0.01
    )

    dataset_config = DatasetConfig(
        n_positive=100,  # Small dataset for demo
        n_negative=100,
        output_dir="data_abcd_demo"
    )

    generator = DatasetGenerator(pattern_config, dataset_config)

    # Generate dataset
    n_pos, n_neg = generator.generate_dataset()

    print(f"\n✓ Created {n_pos + n_neg} images in data_abcd_demo/")


def example_2_generate_single_pattern():
    """Example 2: Generate and visualize single pattern"""
    print("\n" + "="*60)
    print("Example 2: Generate Single Pattern")
    print("="*60 + "\n")

    # Create pattern generator
    generator = ABCDPatternGenerator()

    # Generate bullish ABCD pattern
    df = generator.generate_pattern_ohlcv(
        total_bars=20,
        bullish=True,
        with_pattern=True,
        base_price=100.0
    )

    print("Generated ABCD pattern:")
    print(df.head(10))
    print(f"\nPrice range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

    # Validate pattern points
    A = df['Close'].iloc[0]
    B = df['Close'].iloc[5]
    C = df['Close'].iloc[12]
    D = df['Close'].iloc[-1]

    is_valid, msg = generator.validate_pattern(A, B, C, D)
    print(f"\nPattern validation: {msg}")


def example_3_train_model():
    """Example 3: Train model (requires dataset from example 1)"""
    print("\n" + "="*60)
    print("Example 3: Train Model")
    print("="*60 + "\n")

    if not os.path.exists("data_abcd_demo"):
        print("⚠ Dataset not found. Run example_1_generate_dataset() first!")
        return

    # Setup logging
    setup_logging(log_level="INFO")

    # Configure model (smaller for demo)
    model_config = ModelConfig(
        epochs=5,  # Small number for demo
        batch_size=16,
        early_stopping_patience=3
    )

    # Create trainer
    trainer = Trainer(model_config=model_config)

    # Load datasets
    train_ds, val_ds = trainer.load_dataset("data_abcd_demo")

    # Train model
    history = trainer.train(
        train_ds,
        val_ds,
        model_save_path="models/abcd_detector_demo.h5",
        log_dir="logs"
    )

    print("\n✓ Model trained and saved to models/abcd_detector_demo.h5")


def example_4_detect_pattern():
    """Example 4: Detect pattern in synthetic data"""
    print("\n" + "="*60)
    print("Example 4: Detect Pattern")
    print("="*60 + "\n")

    if not os.path.exists("models/abcd_detector_demo.h5"):
        print("⚠ Model not found. Run example_3_train_model() first!")
        return

    # Load detector
    detector = ABCDDetector("models/abcd_detector_demo.h5")

    # Generate test pattern
    generator = ABCDPatternGenerator()
    df = generator.generate_pattern_ohlcv(
        total_bars=20,
        bullish=True,
        with_pattern=True
    )

    # Detect pattern
    confidence, is_pattern = detector.detect_pattern(df)

    print(f"Detection result:")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Is pattern: {is_pattern}")

    # Try with random data (no pattern)
    df_random = generator.generate_pattern_ohlcv(
        total_bars=20,
        bullish=True,
        with_pattern=False
    )

    confidence_random, is_pattern_random = detector.detect_pattern(df_random)

    print(f"\nRandom data (no pattern):")
    print(f"  Confidence: {confidence_random:.2%}")
    print(f"  Is pattern: {is_pattern_random}")


def example_5_backtest():
    """Example 5: Run backtest on synthetic data"""
    print("\n" + "="*60)
    print("Example 5: Backtest on Synthetic Data")
    print("="*60 + "\n")

    if not os.path.exists("models/abcd_detector_demo.h5"):
        print("⚠ Model not found. Run example_3_train_model() first!")
        return

    # Generate synthetic historical data
    generator = ABCDPatternGenerator()

    # Create 500 bars with several patterns
    dfs = []
    for i in range(10):
        df = generator.generate_pattern_ohlcv(
            total_bars=50,
            bullish=np.random.choice([True, False]),
            with_pattern=np.random.random() > 0.5,  # 50% with pattern
            start_time=datetime.now() - timedelta(hours=500-i*50),
            base_price=100 + np.random.uniform(-10, 10)
        )
        dfs.append(df)

    # Combine into single dataframe
    df = pd.concat(dfs)

    print(f"Generated {len(df)} bars of synthetic data")

    # Load detector
    detector = ABCDDetector("models/abcd_detector_demo.h5")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        position_size_pct=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        detection_threshold=0.6  # Lower threshold for demo
    )

    # Run backtest
    backtest = ABCDBacktest(detector, config)
    result = backtest.run(df, lookback=20, check_interval=10)

    # Print results
    backtest.print_results(result)


def example_6_scan_data():
    """Example 6: Scan data for patterns"""
    print("\n" + "="*60)
    print("Example 6: Scan Data for Patterns")
    print("="*60 + "\n")

    if not os.path.exists("models/abcd_detector_demo.h5"):
        print("⚠ Model not found. Run example_3_train_model() first!")
        return

    # Generate test data
    generator = ABCDPatternGenerator()
    dfs = []

    for i in range(5):
        df = generator.generate_pattern_ohlcv(
            total_bars=30,
            bullish=True,
            with_pattern=i % 2 == 0,  # Patterns at indices 0, 2, 4
            start_time=datetime.now() - timedelta(hours=150-i*30)
        )
        dfs.append(df)

    df = pd.concat(dfs)

    print(f"Scanning {len(df)} bars for patterns...")

    # Load detector
    detector = ABCDDetector("models/abcd_detector_demo.h5")

    # Scan for patterns
    detections = detector.scan_dataframe(
        df,
        window_size=20,
        step=10
    )

    print(f"\nFound {len(detections)} patterns:")
    for i, det in enumerate(detections, 1):
        print(f"\n  Pattern {i}:")
        print(f"    Time: {det['end_time']}")
        print(f"    Confidence: {det['confidence']:.2%}")
        print(f"    Price: {det['price_at_detection']:.2f}")

    # Get statistics
    if detections:
        stats = detector.get_pattern_statistics(detections)
        print(f"\nStatistics:")
        print(f"  Total patterns: {stats['count']}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2%}")
        print(f"  Max confidence: {stats['max_confidence']:.2%}")


def main():
    """Run all examples"""
    print("="*60)
    print("ABCD Pattern Detector - Example Usage")
    print("="*60)

    print("\nThis script demonstrates the complete workflow:")
    print("  1. Generate training dataset")
    print("  2. Generate single pattern")
    print("  3. Train model")
    print("  4. Detect patterns")
    print("  5. Run backtest")
    print("  6. Scan data for patterns")

    # Run examples
    try:
        example_1_generate_dataset()
        example_2_generate_single_pattern()
        example_3_train_model()
        example_4_detect_pattern()
        example_5_backtest()
        example_6_scan_data()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
