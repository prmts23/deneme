"""
Advanced Labeling Methods for ML Trading
Triple Barrier Method and Meta-Labeling
"""

import numpy as np
import pandas as pd
from numba import jit
import warnings
warnings.filterwarnings('ignore')


class TripleBarrierLabeler:
    """
    Implements Triple Barrier Method from "Advances in Financial Machine Learning"
    by Marcos Lopez de Prado
    """

    def __init__(self, config):
        self.config = config

    def create_labels(self, df, direction='both'):
        """
        Create labels using triple barrier method

        Args:
            df: DataFrame with OHLCV data
            direction: 'long', 'short', or 'both'

        Returns:
            DataFrame with labels
        """
        print(f"🏷️  Creating labels using Triple Barrier Method ({direction})...\n")

        df = df.copy()

        # Calculate volatility for dynamic barriers
        if self.config.USE_DYNAMIC_BARRIERS:
            df['volatility'] = self._calculate_volatility(df)
        else:
            df['volatility'] = 1.0  # Will use static barriers

        # Create barriers
        if direction in ['long', 'both']:
            df = self._create_triple_barrier_labels(df, side='long')
            print(f"   Long labels created: {(df['target_long'] == 1).sum():,} signals "
                  f"({(df['target_long'] == 1).sum() / len(df) * 100:.2f}%)")

        if direction in ['short', 'both']:
            df = self._create_triple_barrier_labels(df, side='short')
            print(f"   Short labels created: {(df['target_short'] == 1).sum():,} signals "
                  f"({(df['target_short'] == 1).sum() / len(df) * 100:.2f}%)")

        print()
        return df

    def _calculate_volatility(self, df):
        """Calculate volatility for dynamic barriers (ATR-based)"""
        lookback = self.config.VOLATILITY_LOOKBACK

        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(lookback).mean()

        # Normalize by price
        volatility = atr / df['close']

        return volatility.fillna(volatility.median())

    def _create_triple_barrier_labels(self, df, side='long'):
        """
        Create labels using triple barrier method

        For each bar, we look forward and check:
        1. Upper barrier (take profit)
        2. Lower barrier (stop loss)
        3. Vertical barrier (time limit)

        Label = 1 if TP hit first, 0 otherwise
        """
        barrier_width = self.config.VERTICAL_BARRIER_HOURS

        if side == 'long':
            target_col = 'target_long'
            return_col = 'return_long'
        else:
            target_col = 'target_short'
            return_col = 'return_short'

        df[target_col] = 0
        df[return_col] = 0.0

        # Vectorized approach for performance
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volatility = df['volatility'].values

        n = len(df)

        for i in range(n - barrier_width):
            entry_price = close_prices[i]
            vol = volatility[i]

            # Set barriers based on volatility
            if self.config.USE_DYNAMIC_BARRIERS:
                # Dynamic barriers (ATR-based)
                tp_barrier = entry_price * (1 + vol * 2.0)  # 2x ATR for TP
                sl_barrier = entry_price * (1 - vol * 1.0)  # 1x ATR for SL

                if side == 'short':
                    tp_barrier = entry_price * (1 - vol * 2.0)
                    sl_barrier = entry_price * (1 + vol * 1.0)
            else:
                # Static barriers
                tp_pct = self.config.STATIC_TP_PCT / 100
                sl_pct = self.config.STATIC_SL_PCT / 100

                tp_barrier = entry_price * (1 + tp_pct)
                sl_barrier = entry_price * (1 - sl_pct)

                if side == 'short':
                    tp_barrier = entry_price * (1 - tp_pct)
                    sl_barrier = entry_price * (1 + sl_pct)

            # Look forward
            future_window = slice(i + 1, min(i + 1 + barrier_width, n))
            future_highs = high_prices[future_window]
            future_lows = low_prices[future_window]

            # Check barriers
            if side == 'long':
                # TP hit: high >= tp_barrier
                tp_hit_indices = np.where(future_highs >= tp_barrier)[0]
                # SL hit: low <= sl_barrier
                sl_hit_indices = np.where(future_lows <= sl_barrier)[0]

                if len(tp_hit_indices) > 0 and len(sl_hit_indices) > 0:
                    # Both hit - which one first?
                    if tp_hit_indices[0] <= sl_hit_indices[0]:
                        # TP hit first
                        df.loc[i, target_col] = 1
                        df.loc[i, return_col] = (tp_barrier - entry_price) / entry_price * 100
                    else:
                        # SL hit first
                        df.loc[i, target_col] = 0
                        df.loc[i, return_col] = (sl_barrier - entry_price) / entry_price * 100
                elif len(tp_hit_indices) > 0:
                    # Only TP hit
                    df.loc[i, target_col] = 1
                    df.loc[i, return_col] = (tp_barrier - entry_price) / entry_price * 100
                elif len(sl_hit_indices) > 0:
                    # Only SL hit
                    df.loc[i, target_col] = 0
                    df.loc[i, return_col] = (sl_barrier - entry_price) / entry_price * 100
                else:
                    # Vertical barrier (time limit) - no significant move
                    final_price = close_prices[min(i + barrier_width, n - 1)]
                    pct_return = (final_price - entry_price) / entry_price * 100

                    # Only label as 1 if return exceeds minimum threshold
                    if pct_return >= self.config.MIN_RETURN_THRESHOLD:
                        df.loc[i, target_col] = 1
                    df.loc[i, return_col] = pct_return

            else:  # short
                # TP hit: low <= tp_barrier
                tp_hit_indices = np.where(future_lows <= tp_barrier)[0]
                # SL hit: high >= sl_barrier
                sl_hit_indices = np.where(future_highs >= sl_barrier)[0]

                if len(tp_hit_indices) > 0 and len(sl_hit_indices) > 0:
                    if tp_hit_indices[0] <= sl_hit_indices[0]:
                        df.loc[i, target_col] = 1
                        df.loc[i, return_col] = (entry_price - tp_barrier) / entry_price * 100
                    else:
                        df.loc[i, target_col] = 0
                        df.loc[i, return_col] = (entry_price - sl_barrier) / entry_price * 100
                elif len(tp_hit_indices) > 0:
                    df.loc[i, target_col] = 1
                    df.loc[i, return_col] = (entry_price - tp_barrier) / entry_price * 100
                elif len(sl_hit_indices) > 0:
                    df.loc[i, target_col] = 0
                    df.loc[i, return_col] = (entry_price - sl_barrier) / entry_price * 100
                else:
                    final_price = close_prices[min(i + barrier_width, n - 1)]
                    pct_return = (entry_price - final_price) / entry_price * 100

                    if pct_return >= self.config.MIN_RETURN_THRESHOLD:
                        df.loc[i, target_col] = 1
                    df.loc[i, return_col] = pct_return

        return df

    def add_sample_weights(self, df, target_col='target_long'):
        """
        Add sample weights based on uniqueness and concurrency
        (Advanced concept from Lopez de Prado)
        """
        # For now, simple implementation: weight by return magnitude
        # More sophisticated: weight by label uniqueness over time

        return_col = target_col.replace('target', 'return')

        if return_col in df.columns:
            # Weight by absolute return (higher return = higher weight)
            df['sample_weight'] = abs(df[return_col])

            # Normalize
            df['sample_weight'] = df['sample_weight'] / df['sample_weight'].sum() * len(df)
        else:
            df['sample_weight'] = 1.0

        return df


class MetaLabeler:
    """
    Meta-labeling: Instead of predicting direction, predict whether to trade or not
    Given a primary model's signal, should we take it?
    """

    def __init__(self, config):
        self.config = config

    def create_meta_labels(self, df, primary_signal_col, target_col='target_long'):
        """
        Create meta labels:
        - 1 if primary signal leads to profit
        - 0 if primary signal should be skipped
        """
        df = df.copy()

        meta_target = f'meta_{target_col}'
        return_col = target_col.replace('target', 'return')

        # Meta label: 1 if signal AND profitable, 0 otherwise
        df[meta_target] = (
            (df[primary_signal_col] == 1) &
            (df[return_col] > 0)
        ).astype(int)

        return df


def fractional_differentiation(series, d=0.5, threshold=1e-5):
    """
    Apply fractional differentiation to make series stationary
    while preserving memory (Lopez de Prado technique)

    d: differentiation order (0.5 is common)
    """
    weights = [1.0]
    k = 1

    # Calculate weights
    while abs(weights[-1]) > threshold:
        weight = -weights[-1] * (d - k + 1) / k
        weights.append(weight)
        k += 1

    weights = np.array(weights[::-1])

    # Apply weights
    result = pd.Series(index=series.index, dtype=float)

    for i in range(len(weights), len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i-len(weights):i])

    return result.fillna(0)
