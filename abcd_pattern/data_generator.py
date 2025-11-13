"""
Dataset Generator for ABCD Pattern Detection
Creates labeled image datasets from synthetic patterns
"""
import os
import numpy as np
import pandas as pd
import mplfinance as mpf
from typing import Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from .pattern_generator import ABCDPatternGenerator
from .config import DatasetConfig, PatternConfig

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generator for ABCD pattern image datasets"""

    def __init__(
        self,
        pattern_config: Optional[PatternConfig] = None,
        dataset_config: Optional[DatasetConfig] = None
    ):
        """
        Initialize dataset generator

        Args:
            pattern_config: Pattern generation configuration
            dataset_config: Dataset configuration
        """
        self.pattern_config = pattern_config or PatternConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.pattern_generator = ABCDPatternGenerator(self.pattern_config)

    def save_ohlcv_image(
        self,
        df: pd.DataFrame,
        filepath: str,
        style: str = "charles"
    ) -> None:
        """
        Save OHLCV data as candlestick chart image

        Args:
            df: OHLCV DataFrame
            filepath: Output file path
            style: mplfinance style
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Configure plot style
        mc = mpf.make_marketcolors(
            up='#00ff00',
            down='#ff0000',
            edge='inherit',
            wick='inherit',
            volume='in'
        )

        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=False)

        try:
            mpf.plot(
                df,
                type='candle',
                volume=True,
                style=s,
                savefig=dict(
                    fname=filepath,
                    dpi=self.dataset_config.dpi,
                    bbox_inches='tight',
                    pad_inches=0.1
                ),
                figsize=(8, 6),
                warn_too_much_data=10000
            )
        except Exception as e:
            logger.error(f"Failed to save image {filepath}: {e}")
            raise

    def generate_single_sample(
        self,
        with_pattern: bool,
        idx: int,
        output_dir: str
    ) -> bool:
        """
        Generate single training sample

        Args:
            with_pattern: True for positive sample, False for negative
            idx: Sample index
            output_dir: Output directory

        Returns:
            True if successful
        """
        try:
            # Random parameters
            total_bars = np.random.randint(
                self.pattern_config.min_bars,
                self.pattern_config.max_bars + 1
            )
            bullish = np.random.choice([True, False])

            # Generate OHLCV
            df = self.pattern_generator.generate_pattern_ohlcv(
                total_bars=total_bars,
                bullish=bullish,
                with_pattern=with_pattern
            )

            # Determine subdirectory
            subdir = (self.dataset_config.positive_dir if with_pattern
                     else self.dataset_config.negative_dir)

            # Save image
            filepath = os.path.join(output_dir, subdir, f"{idx:06d}.png")
            self.save_ohlcv_image(df, filepath)

            return True

        except Exception as e:
            logger.error(f"Failed to generate sample {idx}: {e}")
            return False

    def generate_dataset(
        self,
        n_positive: Optional[int] = None,
        n_negative: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Generate complete dataset

        Args:
            n_positive: Number of positive samples (with pattern)
            n_negative: Number of negative samples (without pattern)
            output_dir: Output directory

        Returns:
            (n_positive_created, n_negative_created)
        """
        n_positive = n_positive or self.dataset_config.n_positive
        n_negative = n_negative or self.dataset_config.n_negative
        output_dir = output_dir or self.dataset_config.output_dir

        logger.info(f"Generating dataset: {n_positive} positive, {n_negative} negative")

        # Create directories
        pos_dir = os.path.join(output_dir, self.dataset_config.positive_dir)
        neg_dir = os.path.join(output_dir, self.dataset_config.negative_dir)
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # Generate positive samples
        logger.info("Generating positive samples (with ABCD pattern)...")
        pos_success = 0
        for i in tqdm(range(n_positive), desc="Positive samples"):
            if self.generate_single_sample(True, i, output_dir):
                pos_success += 1

        # Generate negative samples
        logger.info("Generating negative samples (without pattern)...")
        neg_success = 0
        for i in tqdm(range(n_negative), desc="Negative samples"):
            if self.generate_single_sample(False, i, output_dir):
                neg_success += 1

        logger.info(f"Dataset generation complete: {pos_success}/{n_positive} positive, "
                   f"{neg_success}/{n_negative} negative")

        return pos_success, neg_success

    def augment_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augmentation_factor: int = 2
    ) -> int:
        """
        Apply data augmentation to existing dataset

        Args:
            input_dir: Input directory with images
            output_dir: Output directory for augmented images
            augmentation_factor: How many augmented versions per image

        Returns:
            Number of augmented images created
        """
        # TODO: Implement data augmentation
        # - Horizontal flip
        # - Brightness adjustment
        # - Noise injection
        # - Scaling
        logger.warning("Data augmentation not yet implemented")
        return 0

    def validate_dataset(self, dataset_dir: str) -> dict:
        """
        Validate dataset structure and contents

        Args:
            dataset_dir: Dataset directory

        Returns:
            Dictionary with validation results
        """
        pos_dir = os.path.join(dataset_dir, self.dataset_config.positive_dir)
        neg_dir = os.path.join(dataset_dir, self.dataset_config.negative_dir)

        pos_files = list(Path(pos_dir).glob("*.png")) if os.path.exists(pos_dir) else []
        neg_files = list(Path(neg_dir).glob("*.png")) if os.path.exists(neg_dir) else []

        results = {
            "positive_samples": len(pos_files),
            "negative_samples": len(neg_files),
            "total_samples": len(pos_files) + len(neg_files),
            "class_balance": len(pos_files) / (len(pos_files) + len(neg_files) + 1e-10)
        }

        logger.info(f"Dataset validation: {results}")
        return results
