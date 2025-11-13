#!/usr/bin/env python3
"""
Script to generate ABCD pattern training dataset
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abcd_pattern import (
    DatasetGenerator,
    PatternConfig,
    DatasetConfig,
    setup_logging
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ABCD pattern training dataset'
    )

    parser.add_argument(
        '--n-positive',
        type=int,
        default=1000,
        help='Number of positive samples (with ABCD pattern)'
    )

    parser.add_argument(
        '--n-negative',
        type=int,
        default=1000,
        help='Number of negative samples (without pattern)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data_abcd',
        help='Output directory for dataset'
    )

    parser.add_argument(
        '--min-bars',
        type=int,
        default=7,
        help='Minimum bars in pattern'
    )

    parser.add_argument(
        '--max-bars',
        type=int,
        default=30,
        help='Maximum bars in pattern'
    )

    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.01,
        help='Noise level for pattern generation'
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

    # Configure pattern generation
    pattern_config = PatternConfig(
        min_bars=args.min_bars,
        max_bars=args.max_bars,
        noise_level=args.noise_level
    )

    # Configure dataset
    dataset_config = DatasetConfig(
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        output_dir=args.output_dir
    )

    # Create generator
    generator = DatasetGenerator(
        pattern_config=pattern_config,
        dataset_config=dataset_config
    )

    print("="*60)
    print("ABCD Pattern Dataset Generator")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Positive samples: {args.n_positive}")
    print(f"  Negative samples: {args.n_negative}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Pattern bars: {args.min_bars}-{args.max_bars}")
    print(f"  Noise level: {args.noise_level}")
    print("\nStarting generation...\n")

    # Generate dataset
    n_pos, n_neg = generator.generate_dataset()

    print("\n" + "="*60)
    print("Dataset Generation Complete")
    print("="*60)
    print(f"\nCreated:")
    print(f"  Positive samples: {n_pos}/{args.n_positive}")
    print(f"  Negative samples: {n_neg}/{args.n_negative}")
    print(f"  Total: {n_pos + n_neg}")
    print(f"\nDataset saved to: {args.output_dir}")

    # Validate dataset
    print("\nValidating dataset...")
    results = generator.validate_dataset(args.output_dir)
    print(f"  Positive samples: {results['positive_samples']}")
    print(f"  Negative samples: {results['negative_samples']}")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Class balance: {results['class_balance']:.2%}")

    print("\n✓ Dataset ready for training!")


if __name__ == '__main__':
    main()
