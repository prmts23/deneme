"""
Utility functions for ABCD Pattern Detection
"""
import logging
import sys
import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path
import json


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Custom log format string

    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")

    return logger


def load_ohlcv_data(
    filepath: str,
    date_column: str = "Date",
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file

    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        parse_dates: Parse dates

    Returns:
        OHLCV DataFrame with datetime index
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading OHLCV data from {filepath}")

    # Determine file format
    file_ext = Path(filepath).suffix.lower()

    if file_ext == '.csv':
        df = pd.read_csv(filepath)
    elif file_ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif file_ext == '.json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Set datetime index
    if date_column in df.columns:
        if parse_dates:
            df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)

    # Validate OHLCV columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure proper dtypes
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df.dropna(inplace=True)

    # Sort by date
    df.sort_index(inplace=True)

    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def save_results_to_json(results: dict, filepath: str) -> None:
    """
    Save results dictionary to JSON file

    Args:
        results: Results dictionary
        filepath: Output file path
    """
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    # Recursively convert types
    def convert_dict(d):
        return {
            k: convert_dict(v) if isinstance(v, dict)
            else [convert_dict(i) if isinstance(i, dict) else convert_types(i) for i in v]
            if isinstance(v, list)
            else convert_types(v)
            for k, v in d.items()
        }

    converted_results = convert_dict(results)

    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)

    logger = logging.getLogger(__name__)
    logger.info(f"Results saved to {filepath}")


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with added indicators
    """
    df = df.copy()

    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    return df


def validate_ohlcv_data(df: pd.DataFrame, raise_error: bool = True) -> bool:
    """
    Validate OHLCV data integrity

    Args:
        df: OHLCV DataFrame
        raise_error: Raise error if validation fails

    Returns:
        True if valid
    """
    issues = []

    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

    # Check High >= Low
    invalid_hl = df[df['High'] < df['Low']]
    if len(invalid_hl) > 0:
        issues.append(f"High < Low in {len(invalid_hl)} rows")

    # Check High >= Open, Close
    invalid_h_open = df[df['High'] < df['Open']]
    invalid_h_close = df[df['High'] < df['Close']]
    if len(invalid_h_open) > 0 or len(invalid_h_close) > 0:
        issues.append(f"High < Open/Close in {len(invalid_h_open) + len(invalid_h_close)} rows")

    # Check Low <= Open, Close
    invalid_l_open = df[df['Low'] > df['Open']]
    invalid_l_close = df[df['Low'] > df['Close']]
    if len(invalid_l_open) > 0 or len(invalid_l_close) > 0:
        issues.append(f"Low > Open/Close in {len(invalid_l_open) + len(invalid_l_close)} rows")

    # Check for negative prices
    negative_prices = df[(df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)]
    if len(negative_prices) > 0:
        issues.append(f"Negative or zero prices in {len(negative_prices)} rows")

    # Check for negative volume
    negative_volume = df[df['Volume'] < 0]
    if len(negative_volume) > 0:
        issues.append(f"Negative volume in {len(negative_volume)} rows")

    if issues:
        error_msg = "OHLCV validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
        if raise_error:
            raise ValueError(error_msg)
        else:
            logger = logging.getLogger(__name__)
            logger.warning(error_msg)
            return False

    return True


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe

    Args:
        df: OHLCV DataFrame
        timeframe: Target timeframe (e.g., '1H', '4H', '1D')

    Returns:
        Resampled DataFrame
    """
    resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Drop rows with NaN (incomplete periods)
    resampled.dropna(inplace=True)

    return resampled


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> tuple:
    """
    Split data into train, validation, and test sets

    Args:
        df: DataFrame to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger = logging.getLogger(__name__)
    logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df
