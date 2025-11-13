"""
Configuration module for ABCD Pattern Detector
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import os


@dataclass
class PatternConfig:
    """Configuration for ABCD pattern generation"""
    min_bars: int = 7
    max_bars: int = 30
    noise_level: float = 0.01

    # Fibonacci retracement levels
    min_retracement: float = 0.382
    max_retracement: float = 0.886

    # ABCD ratio (CD/AB)
    min_abcd_ratio: float = 0.9
    max_abcd_ratio: float = 1.618

    # AB leg size
    min_ab_size: float = 1.0
    max_ab_size: float = 2.0


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    n_positive: int = 1000
    n_negative: int = 1000
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1

    image_size: Tuple[int, int] = (224, 224)
    dpi: int = 100

    output_dir: str = "data_abcd"
    positive_dir: str = "abcd"
    negative_dir: str = "none"

    def __post_init__(self):
        """Validate split ratios"""
        total = self.train_split + self.validation_split + self.test_split
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class ModelConfig:
    """Configuration for CNN model"""
    input_shape: Tuple[int, int, int] = (224, 224, 3)

    # Convolutional layers
    conv_filters: Tuple[int, ...] = (32, 64, 128, 256)
    conv_kernel_size: int = 3
    pool_size: int = 2

    # Dense layers
    dense_units: Tuple[int, ...] = (256, 128)
    dropout_rate: float = 0.5

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10

    # Optimizer
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    metrics: Tuple[str, ...] = ("accuracy", "precision", "recall", "auc")


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Position sizing
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1  # 10% of capital per trade
    max_positions: int = 3

    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit

    # Pattern detection threshold
    detection_threshold: float = 0.7  # Model confidence threshold

    # Trading hours
    start_hour: Optional[int] = None
    end_hour: Optional[int] = None

    # Commission
    commission_pct: float = 0.001  # 0.1% commission


@dataclass
class Config:
    """Main configuration container"""
    pattern: PatternConfig = field(default_factory=PatternConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Paths
    model_save_path: str = "models/abcd_detector.h5"
    log_dir: str = "logs"

    # Random seed for reproducibility
    random_seed: int = 42

    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.dataset.output_dir, exist_ok=True)


# Global config instance
config = Config()
