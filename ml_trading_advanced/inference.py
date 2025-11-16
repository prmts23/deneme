"""
Inference Module - Use trained models for predictions

This module can be integrated into:
- Freqtrade strategies
- MT5 expert advisors
- Custom trading bots
"""

import os
import pickle
import numpy as np
import pandas as pd
from feature_engineering import AdvancedFeatureEngineer
import config


class TradingPredictor:
    """
    Wrapper class for making predictions with trained models
    """

    def __init__(self, model_dir='models_advanced'):
        """
        Initialize predictor

        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.feature_engineer = AdvancedFeatureEngineer(config)

        # Load models and scalers
        self.model_long = None
        self.scaler_long = None
        self.model_short = None
        self.scaler_short = None
        self.features = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained models, scalers, and feature list"""

        # Load features
        feature_path = os.path.join(self.model_dir, 'features.txt')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(self.features)} features")
        else:
            raise FileNotFoundError(f"Feature list not found: {feature_path}")

        # Try to load best models (try different model types)
        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            # LONG
            long_model_path = os.path.join(self.model_dir, f'model_long_{model_type}.pkl')
            long_scaler_path = os.path.join(self.model_dir, f'scaler_long_{model_type}.pkl')

            if os.path.exists(long_model_path) and self.model_long is None:
                with open(long_model_path, 'rb') as f:
                    self.model_long = pickle.load(f)
                with open(long_scaler_path, 'rb') as f:
                    self.scaler_long = pickle.load(f)
                print(f"✓ Loaded LONG {model_type} model")

            # SHORT
            short_model_path = os.path.join(self.model_dir, f'model_short_{model_type}.pkl')
            short_scaler_path = os.path.join(self.model_dir, f'scaler_short_{model_type}.pkl')

            if os.path.exists(short_model_path) and self.model_short is None:
                with open(short_model_path, 'rb') as f:
                    self.model_short = pickle.load(f)
                with open(short_scaler_path, 'rb') as f:
                    self.scaler_short = pickle.load(f)
                print(f"✓ Loaded SHORT {model_type} model")

        if self.model_long is None:
            print("⚠️  No LONG model found")
        if self.model_short is None:
            print("⚠️  No SHORT model found")

    def prepare_features(self, df):
        """
        Prepare features from raw OHLCV data

        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with engineered features
        """
        # Ensure column names are lowercase
        df = df.copy()
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        # Engineer features
        df = self.feature_engineer.engineer_all_features(df)

        return df

    def predict(self, df, side='long', threshold=0.5):
        """
        Make predictions on new data

        Args:
            df: DataFrame with OHLCV data (can be raw or pre-processed)
            side: 'long' or 'short'
            threshold: Probability threshold for classification

        Returns:
            Dictionary with predictions and probabilities
        """
        # Check if features are already engineered
        if not all(feat in df.columns for feat in self.features[:5]):
            # Need to engineer features
            df = self.prepare_features(df)

        # Extract features
        X = df[self.features].copy()

        # Handle missing/inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Select model and scaler
        if side == 'long':
            if self.model_long is None:
                raise ValueError("LONG model not loaded")
            model = self.model_long
            scaler = self.scaler_long
        elif side == 'short':
            if self.model_short is None:
                raise ValueError("SHORT model not loaded")
            model = self.model_short
            scaler = self.scaler_short
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'long' or 'short'")

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        probabilities = model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'side': side,
            'threshold': threshold
        }

    def predict_latest(self, df, side='both', long_threshold=0.5, short_threshold=0.5):
        """
        Predict on the latest bar only (useful for live trading)

        Args:
            df: DataFrame with sufficient history for feature engineering
            side: 'long', 'short', or 'both'
            long_threshold: Probability threshold for LONG signals
            short_threshold: Probability threshold for SHORT signals

        Returns:
            Dictionary with signal information
        """
        # Engineer features
        if not all(feat in df.columns for feat in self.features[:5]):
            df = self.prepare_features(df)

        # Get last valid row
        last_idx = df.index[-1]

        result = {
            'timestamp': df.loc[last_idx, 'timestamp'] if 'timestamp' in df.columns else None,
            'close': df.loc[last_idx, 'close'],
            'signal': 0,  # 0 = no signal, 1 = long, -1 = short
            'long_prob': 0.0,
            'short_prob': 0.0
        }

        # LONG prediction
        if side in ['long', 'both'] and self.model_long is not None:
            pred_long = self.predict(df, side='long', threshold=long_threshold)
            result['long_prob'] = pred_long['probabilities'][-1]

            if pred_long['predictions'][-1] == 1:
                result['signal'] = 1

        # SHORT prediction
        if side in ['short', 'both'] and self.model_short is not None:
            pred_short = self.predict(df, side='short', threshold=short_threshold)
            result['short_prob'] = pred_short['probabilities'][-1]

            if pred_short['predictions'][-1] == 1:
                # If both long and short, prioritize higher probability
                if result['signal'] == 1:
                    if result['short_prob'] > result['long_prob']:
                        result['signal'] = -1
                else:
                    result['signal'] = -1

        return result


def example_usage():
    """
    Example of how to use the predictor
    """
    # Load your data
    df = pd.read_csv('/content/AVAXUSDT_5m_ALL_YEARS.csv')

    # Initialize predictor
    predictor = TradingPredictor(model_dir='models_advanced')

    # Get prediction for latest bar
    signal = predictor.predict_latest(df, side='both', long_threshold=0.6, short_threshold=0.6)

    print("\n" + "=" * 80)
    print("LATEST PREDICTION")
    print("=" * 80)
    print(f"Timestamp: {signal['timestamp']}")
    print(f"Close: {signal['close']:.2f}")
    print(f"LONG probability: {signal['long_prob']:.3f}")
    print(f"SHORT probability: {signal['short_prob']:.3f}")

    if signal['signal'] == 1:
        print(f"\n🟢 SIGNAL: LONG")
    elif signal['signal'] == -1:
        print(f"\n🔴 SIGNAL: SHORT")
    else:
        print(f"\n⚪ SIGNAL: NEUTRAL (No trade)")

    print("=" * 80)


if __name__ == '__main__':
    example_usage()
