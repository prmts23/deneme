"""
Advanced ML Trading Strategy - Configuration
"""

import os

# ============================================================
# DATA CONFIGURATION
# ============================================================

# ⚠️  IMPORTANT: Update this path to your CSV file!
# Examples:
#   Linux/Mac: "/home/user/data/AVAXUSDT_5m.csv"
#   Windows: "C:/Users/YourName/data/AVAXUSDT_5m.csv"
#   Google Colab: "/content/AVAXUSDT_5m_ALL_YEARS.csv"
#
# CSV must have columns: timestamp, open, high, low, close, volume

DATA_PATH = "/content/AVAXUSDT_5m_ALL_YEARS.csv"  # ⚠️  UPDATE THIS!

SYMBOL = "AVAXUSDT"
TIMEFRAME = "5m"

# ============================================================
# LABELING CONFIGURATION (Triple Barrier Method)
# ============================================================

# Volatility-based barriers
BARRIER_METHOD = "triple"  # "triple" or "simple"
VERTICAL_BARRIER_HOURS = 2  # Maximum holding period (in candles for 5m = 2 hours)

# Dynamic barriers based on volatility
USE_DYNAMIC_BARRIERS = True
VOLATILITY_LOOKBACK = 20  # bars for ATR calculation

# Static barriers (used if USE_DYNAMIC_BARRIERS = False)
STATIC_TP_PCT = 1.5  # %
STATIC_SL_PCT = 1.0  # %

# Minimum return threshold for labeling as 1
MIN_RETURN_THRESHOLD = 0.3  # % minimum profit to consider as signal

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Time features
USE_TIME_FEATURES = True

# Volume features
USE_VOLUME_FEATURES = True
VOLUME_PROFILE_BINS = 10

# Market regime features
USE_REGIME_FEATURES = True
REGIME_LOOKBACK = 50

# Fractal features
USE_FRACTAL_FEATURES = True
FRACTAL_PERIODS = [5, 13, 21]

# Orderflow/microstructure
USE_MICROSTRUCTURE_FEATURES = True

# Volatility clustering
USE_VOLATILITY_FEATURES = True
GARCH_LAG = 5

# Technical indicators
USE_TA_FEATURES = True

# Feature selection
FEATURE_SELECTION_METHOD = "shap"  # "shap", "permutation", "recursive"
TOP_N_FEATURES = 50  # Final feature count

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Models to train
MODELS = {
    'xgboost': True,
    'lightgbm': True,
    'catboost': True,
    'random_forest': False,  # Usually worse than boosting
}

# Hyperparameter optimization
USE_OPTUNA = True
OPTUNA_TRIALS = 100
OPTUNA_CV_FOLDS = 3

# Class imbalance
USE_SMOTE = False  # Be careful with time series
USE_CLASS_WEIGHTS = True

# Probability calibration
USE_CALIBRATION = True
CALIBRATION_METHOD = "isotonic"  # "isotonic" or "sigmoid"

# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

# Train/test split
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.15

# Time series CV
USE_TIME_SERIES_CV = True
N_SPLITS = 5
GAP_SIZE = 12  # bars between train and test (avoid data leakage)

# Walk-forward validation
USE_WALK_FORWARD = True
WALK_FORWARD_WINDOW = 10000  # training window size
WALK_FORWARD_STEP = 2000     # step size

# ============================================================
# PERFORMANCE METRICS
# ============================================================

# Metrics to track
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision',
    'mcc',  # Matthews correlation coefficient
]

# Probability thresholds to test
PROBABILITY_THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

OUTPUT_DIR = "models_advanced"
SAVE_PREDICTIONS = True
SAVE_FEATURE_IMPORTANCE = True
SAVE_SHAP_VALUES = True
GENERATE_PLOTS = True

# ============================================================
# RUNTIME
# ============================================================

RANDOM_STATE = 42
N_JOBS = -1  # Use all CPU cores
VERBOSE = 1
