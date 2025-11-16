"""
Advanced ML Trading Strategy Package

A professional, institutional-grade machine learning trading system with:
- Triple Barrier Method labeling
- 200+ advanced features
- Hyperparameter optimization
- Walk-forward validation
- Production-ready inference

Author: Advanced ML Trading Team
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Advanced ML Trading Team'

from .feature_engineering import AdvancedFeatureEngineer
from .labeling import TripleBarrierLabeler, MetaLabeler, fractional_differentiation
from .model_training import ModelTrainer
from .inference import TradingPredictor

__all__ = [
    'AdvancedFeatureEngineer',
    'TripleBarrierLabeler',
    'MetaLabeler',
    'fractional_differentiation',
    'ModelTrainer',
    'TradingPredictor',
]
