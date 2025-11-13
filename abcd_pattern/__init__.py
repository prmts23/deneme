"""
ABCD Pattern Detection System
Professional toolkit for detecting harmonic ABCD patterns in financial data
"""

__version__ = "1.0.0"
__author__ = "ABCD Pattern Detection Team"

from .config import (
    Config,
    PatternConfig,
    DatasetConfig,
    ModelConfig,
    BacktestConfig,
    config
)
from .pattern_generator import ABCDPatternGenerator
from .data_generator import DatasetGenerator
from .model import ABCDPatternCNN
from .trainer import Trainer
from .detector import ABCDDetector
from .utils import setup_logging, load_ohlcv_data

__all__ = [
    # Config
    'Config',
    'PatternConfig',
    'DatasetConfig',
    'ModelConfig',
    'BacktestConfig',
    'config',

    # Core classes
    'ABCDPatternGenerator',
    'DatasetGenerator',
    'ABCDPatternCNN',
    'Trainer',
    'ABCDDetector',

    # Utilities
    'setup_logging',
    'load_ohlcv_data',
]
