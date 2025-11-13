#!/usr/bin/env python3
"""
Comprehensive Harmonic Pattern Detection Strategy for Freqtrade
================================================================

Scott Carney standard harmonic patterns:
- Gartley, Bat, Crab, Butterfly, Cypher, Shark
- Vectorized implementation without loops
- Bullish/bearish symmetry
- D-point price control

Usage:
    freqtrade trade --strategy HarmonicPatternsStrategy
"""

import logging
import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta

logger = logging.getLogger(__name__)


class SwingDetector:
    """Enhanced swing point detection using SciPy signal processing."""

    @staticmethod
    def detect_swings(dataframe: pd.DataFrame, prominence_factor: float = 0.001,
                     distance: int = 6) -> pd.DataFrame:
        """Detect swing highs and lows using optimized SciPy peaks method."""
        df = dataframe.copy()

        try:
            high_prices = df['high'].values
            low_prices = df['low'].values

            # Validate data
            if len(high_prices) < distance * 2:
                df['swing_high'] = False
                df['swing_low'] = False
                return df

            price_range = np.max(high_prices) - np.min(low_prices)
            if price_range == 0:
                df['swing_high'] = False
                df['swing_low'] = False
                return df

            prominence_threshold = price_range * prominence_factor

            high_peaks, _ = signal.find_peaks(
                high_prices,
                prominence=prominence_threshold,
                distance=distance
            )

            low_peaks, _ = signal.find_peaks(
                -low_prices,
                prominence=prominence_threshold,
                distance=distance
            )

            df['swing_high'] = False
            df['swing_low'] = False

            if len(high_peaks) > 0:
                df.loc[df.index[high_peaks], 'swing_high'] = True
            if len(low_peaks) > 0:
                df.loc[df.index[low_peaks], 'swing_low'] = True

        except Exception as e:
            logger.error(f"Swing detection failed: {e}")
            df['swing_high'] = False
            df['swing_low'] = False

        return df


class HarmonicPatternSpecs:
    """Scott Carney harmonic pattern specifications."""

    @staticmethod
    def get_pattern_specs() -> Dict[str, Dict]:
        """Get Scott Carney standard specifications for all harmonic patterns."""
        return {
            'gartley': {
                'name': 'Gartley',
                'ab_xa_range': (0.618, 0.618),
                'bc_ab_range': (0.382, 0.886),
                'cd_ab_range': (1.272, 1.618),
                'xd_xa_range': (0.786, 0.786)
            },
            'bat': {
                'name': 'Bat',
                'ab_xa_range': (0.382, 0.5),
                'bc_ab_range': (0.382, 0.886),
                'cd_ab_range': (1.618, 2.618),
                'xd_xa_range': (0.886, 0.886)
            },
            'crab': {
                'name': 'Crab',
                'ab_xa_range': (0.382, 0.618),
                'bc_ab_range': (0.382, 0.886),
                'cd_ab_range': (2.24, 3.618),
                'xd_xa_range': (1.618, 1.618)
            },
            'butterfly': {
                'name': 'Butterfly',
                'ab_xa_range': (0.786, 0.786),
                'bc_ab_range': (0.382, 0.886),
                'cd_ab_range': (1.618, 2.618),
                'xd_xa_range': (1.27, 1.27)
            },
            'cypher': {
                'name': 'Cypher',
                'ab_xa_range': (0.382, 0.786),
                'bc_ab_range': (1.272, 1.414),
                'cd_bc_range': (0.786, 0.786),
                'xd_xa_range': (0.786, 0.786)
            },
            'shark': {
                'name': 'Shark',
                'ab_xa_range': (0.382, 0.618),
                'bc_ab_range': (1.13, 1.618),
                'cd_ab_range': (1.618, 2.214),
                'xd_xa_range': (0.886, 1.13)
            }
        }


class VectorizedHarmonicDetector:
    """Vectorized detector for all harmonic patterns."""

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.pattern_specs = HarmonicPatternSpecs.get_pattern_specs()

    def generate_xabcd_combinations(self, swing_highs: pd.DataFrame, swing_lows: pd.DataFrame,
                                  pattern_type: str = 'bullish', max_patterns: int = 50) -> pd.DataFrame:
        """Generate XABCD combinations using vectorized operations."""
        if len(swing_highs) < 2 or len(swing_lows) < 3:
            return pd.DataFrame()

        swing_highs = swing_highs.reset_index(drop=False)
        swing_lows = swing_lows.reset_index(drop=False)

        # Keep original index for reference
        swing_highs['orig_idx'] = swing_highs.index
        swing_lows['orig_idx'] = swing_lows.index

        if pattern_type == 'bullish':
            x_df = swing_lows[['index', 'low']].rename(columns={'low': 'x_price', 'index': 'x_idx'}).tail(8)
            a_df = swing_highs[['index', 'high']].rename(columns={'high': 'a_price', 'index': 'a_idx'}).tail(12)
            b_df = swing_lows[['index', 'low']].rename(columns={'low': 'b_price', 'index': 'b_idx'}).tail(12)
            c_df = swing_highs[['index', 'high']].rename(columns={'high': 'c_price', 'index': 'c_idx'}).tail(12)
            d_df = swing_lows[['index', 'low']].rename(columns={'low': 'd_price', 'index': 'd_idx'}).tail(12)
        else:
            x_df = swing_highs[['index', 'high']].rename(columns={'high': 'x_price', 'index': 'x_idx'}).tail(8)
            a_df = swing_lows[['index', 'low']].rename(columns={'low': 'a_price', 'index': 'a_idx'}).tail(12)
            b_df = swing_highs[['index', 'high']].rename(columns={'high': 'b_price', 'index': 'b_idx'}).tail(12)
            c_df = swing_lows[['index', 'low']].rename(columns={'low': 'c_price', 'index': 'c_idx'}).tail(12)
            d_df = swing_highs[['index', 'high']].rename(columns={'high': 'd_price', 'index': 'd_idx'}).tail(12)

        x_df['merge_key'] = 1
        a_df['merge_key'] = 1
        b_df['merge_key'] = 1
        c_df['merge_key'] = 1
        d_df['merge_key'] = 1

        try:
            xa = pd.merge(x_df, a_df, on='merge_key', suffixes=('', ''))
            xa_valid = xa[xa['a_idx'] > xa['x_idx']]

            if xa_valid.empty:
                return pd.DataFrame()

            xab = pd.merge(xa_valid, b_df, on='merge_key', suffixes=('', ''))
            xab_valid = xab[xab['b_idx'] > xab['a_idx']]

            if xab_valid.empty:
                return pd.DataFrame()

            xabc = pd.merge(xab_valid, c_df, on='merge_key', suffixes=('', ''))
            xabc_valid = xabc[xabc['c_idx'] > xabc['b_idx']]

            if xabc_valid.empty:
                return pd.DataFrame()

            xabcd = pd.merge(xabc_valid, d_df, on='merge_key', suffixes=('', ''))
            xabcd_valid = xabcd[xabcd['d_idx'] > xabcd['c_idx']]

            if len(xabcd_valid) > max_patterns:
                xabcd_valid = xabcd_valid.nlargest(max_patterns, 'd_idx')

            result_columns = ['x_idx', 'a_idx', 'b_idx', 'c_idx', 'd_idx',
                            'x_price', 'a_price', 'b_price', 'c_price', 'd_price']

            return xabcd_valid[result_columns].copy()

        except Exception as e:
            logger.error(f"XABCD combination generation failed: {e}")
            return pd.DataFrame()

    def calculate_pattern_ratios(self, combinations_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required ratios for pattern matching."""
        if combinations_df.empty:
            return combinations_df

        try:
            result_df = combinations_df.copy()

            xa_moves = np.abs(result_df['a_price'].values - result_df['x_price'].values)
            ab_moves = np.abs(result_df['b_price'].values - result_df['a_price'].values)
            bc_moves = np.abs(result_df['c_price'].values - result_df['b_price'].values)
            cd_moves = np.abs(result_df['d_price'].values - result_df['c_price'].values)
            xd_moves = np.abs(result_df['d_price'].values - result_df['x_price'].values)

            # Avoid division by zero
            xa_moves = np.where(xa_moves == 0, 1e-10, xa_moves)
            ab_moves = np.where(ab_moves == 0, 1e-10, ab_moves)
            bc_moves = np.where(bc_moves == 0, 1e-10, bc_moves)

            result_df['ab_xa'] = ab_moves / xa_moves
            result_df['bc_ab'] = bc_moves / ab_moves
            result_df['cd_ab'] = cd_moves / ab_moves
            result_df['cd_bc'] = cd_moves / bc_moves
            result_df['xd_xa'] = xd_moves / xa_moves

            return result_df

        except Exception as e:
            logger.error(f"Ratio calculation failed: {e}")
            return combinations_df

    def validate_pattern(self, combinations_df: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """Validate combinations against specific pattern criteria."""
        if combinations_df.empty or pattern_name not in self.pattern_specs:
            return pd.DataFrame()

        spec = self.pattern_specs[pattern_name]
        valid_combinations = []

        for _, combo in combinations_df.iterrows():
            score = 0
            max_score = 4

            # Validate AB/XA ratio
            ab_xa = combo['ab_xa']
            ab_range = spec['ab_xa_range']
            if ab_range[0] - self.tolerance <= ab_xa <= ab_range[1] + self.tolerance:
                score += 1

            # Validate BC/AB ratio
            bc_ab = combo['bc_ab']
            bc_range = spec['bc_ab_range']
            if bc_range[0] - self.tolerance <= bc_ab <= bc_range[1] + self.tolerance:
                score += 1

            # Validate CD ratio (either CD/AB or CD/BC depending on pattern)
            if 'cd_ab_range' in spec:
                cd_ab = combo['cd_ab']
                cd_range = spec['cd_ab_range']
                if cd_range[0] - self.tolerance <= cd_ab <= cd_range[1] + self.tolerance:
                    score += 1
            elif 'cd_bc_range' in spec:
                cd_bc = combo['cd_bc']
                cd_range = spec['cd_bc_range']
                if cd_range[0] - self.tolerance <= cd_bc <= cd_range[1] + self.tolerance:
                    score += 1

            # Validate XD/XA ratio
            xd_xa = combo['xd_xa']
            xd_range = spec['xd_xa_range']
            if xd_range[0] - self.tolerance <= xd_xa <= xd_range[1] + self.tolerance:
                score += 1

            # Accept patterns with score >= 3
            if score >= 3:
                combo_dict = combo.to_dict()
                combo_dict['pattern_name'] = pattern_name
                combo_dict['pattern_score'] = score
                valid_combinations.append(combo_dict)

        return pd.DataFrame(valid_combinations) if valid_combinations else pd.DataFrame()


class HarmonicPatternsStrategy(IStrategy):
    """
    Freqtrade strategy using Scott Carney harmonic patterns.

    Buy signals: Bullish patterns at D-point
    Sell signals: Bearish patterns at D-point or exit conditions
    """

    INTERFACE_VERSION = 3

    # Strategy settings
    minimal_roi = {
        "0": 0.50  # 50% profit target
    }

    stoploss = -0.05  # 5% stoploss

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    can_short = True

    # Process only new candles
    process_only_new_candles = True

    # Detection parameters
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperparameters
    pattern_tolerance = 0.05
    prominence_factor = 0.001
    swing_distance = 6
    min_pattern_score = 3
    pattern_detection_lookback = 100  # Only analyze last N candles

    def informative_pairs(self):
        """Additional pairs for analysis."""
        return []

    def __init__(self, config: dict):
        super().__init__(config)
        self.swing_detector = SwingDetector()
        self.harmonic_detector = VectorizedHarmonicDetector(tolerance=self.pattern_tolerance)
        self.pattern_names = ['gartley', 'bat', 'crab', 'butterfly', 'cypher', 'shark']
        self.active_patterns = {}
        self.last_pattern_check = {}

    def detect_all_patterns(self, dataframe: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Detect all harmonic patterns."""
        # Only analyze recent data for performance
        lookback = min(self.pattern_detection_lookback, len(dataframe))
        df_recent = dataframe.tail(lookback).copy()

        df_with_swings = self.swing_detector.detect_swings(
            df_recent,
            prominence_factor=self.prominence_factor,
            distance=self.swing_distance
        )

        swing_highs = df_with_swings[df_with_swings['swing_high']].copy()
        swing_lows = df_with_swings[df_with_swings['swing_low']].copy()

        if len(swing_highs) < 2 or len(swing_lows) < 3:
            return {name: {'bullish': pd.DataFrame(), 'bearish': pd.DataFrame()}
                   for name in self.pattern_names}

        results = {}

        for pattern_name in self.pattern_names:
            bullish_combinations = self.harmonic_detector.generate_xabcd_combinations(
                swing_highs, swing_lows, 'bullish', max_patterns=30
            )

            bearish_combinations = self.harmonic_detector.generate_xabcd_combinations(
                swing_highs, swing_lows, 'bearish', max_patterns=30
            )

            if not bullish_combinations.empty:
                bullish_combinations = self.harmonic_detector.calculate_pattern_ratios(bullish_combinations)
                bullish_patterns = self.harmonic_detector.validate_pattern(bullish_combinations, pattern_name)
            else:
                bullish_patterns = pd.DataFrame()

            if not bearish_combinations.empty:
                bearish_combinations = self.harmonic_detector.calculate_pattern_ratios(bearish_combinations)
                bearish_patterns = self.harmonic_detector.validate_pattern(bearish_combinations, pattern_name)
            else:
                bearish_patterns = pd.DataFrame()

            if not bullish_patterns.empty:
                bullish_patterns['pattern_type'] = 'bullish'
            if not bearish_patterns.empty:
                bearish_patterns['pattern_type'] = 'bearish'

            results[pattern_name] = {
                'bullish': bullish_patterns,
                'bearish': bearish_patterns
            }

        return results

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add technical indicators."""

        # ATR for position sizing and volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # RSI for confirmation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Volume confirmation
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)

        # Initialize signal columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals based on harmonic patterns."""

        try:
            # Detect patterns
            all_patterns = self.detect_all_patterns(dataframe)

            # Get current index (last candle)
            current_idx = len(dataframe) - 1

            # Check bullish patterns (buy signals)
            for pattern_name in self.pattern_names:
                bullish_patterns = all_patterns[pattern_name]['bullish']

                if not bullish_patterns.empty:
                    for _, pattern in bullish_patterns.iterrows():
                        d_idx = int(pattern['d_idx'])

                        # Check if D-point is recent (within 3 candles from end)
                        # Need to map back to original dataframe index
                        lookback = min(self.pattern_detection_lookback, len(dataframe))
                        offset = len(dataframe) - lookback
                        actual_d_idx = d_idx + offset

                        if current_idx - 3 <= actual_d_idx <= current_idx:
                            pattern_score = pattern['pattern_score']

                            # Generate buy signal with confirmation
                            if pattern_score >= self.min_pattern_score:
                                # Additional confirmation: RSI not overbought
                                if dataframe.loc[dataframe.index[actual_d_idx], 'rsi'] < 70:
                                    dataframe.loc[dataframe.index[actual_d_idx:current_idx+1], 'enter_long'] = 1
                                    logger.info(f"LONG signal: {pattern_name.upper()} (score: {pattern_score})")

            # Check bearish patterns (short signals) if shorting enabled
            if self.can_short:
                for pattern_name in self.pattern_names:
                    bearish_patterns = all_patterns[pattern_name]['bearish']

                    if not bearish_patterns.empty:
                        for _, pattern in bearish_patterns.iterrows():
                            d_idx = int(pattern['d_idx'])

                            lookback = min(self.pattern_detection_lookback, len(dataframe))
                            offset = len(dataframe) - lookback
                            actual_d_idx = d_idx + offset

                            if current_idx - 3 <= actual_d_idx <= current_idx:
                                pattern_score = pattern['pattern_score']

                                if pattern_score >= self.min_pattern_score:
                                    # Additional confirmation: RSI not oversold
                                    if dataframe.loc[dataframe.index[actual_d_idx], 'rsi'] > 30:
                                        dataframe.loc[dataframe.index[actual_d_idx:current_idx+1], 'enter_short'] = 1
                                        logger.info(f"SHORT signal: {pattern_name.upper()} (score: {pattern_score})")

        except Exception as e:
            logger.error(f"Entry signal generation failed: {e}", exc_info=True)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals."""

        # RSI extreme levels for exits
        dataframe.loc[
            (dataframe['rsi'] > 80),
            'exit_long'
        ] = 1

        dataframe.loc[
            (dataframe['rsi'] < 20),
            'exit_short'
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each trade.
        """
        return 1.0

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic based on ATR.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            atr = last_candle['atr']

            # Use ATR-based stoploss (2x ATR)
            if trade.is_short:
                atr_stop = (current_rate + 2 * atr - trade.open_rate) / trade.open_rate
            else:
                atr_stop = (current_rate - 2 * atr - trade.open_rate) / trade.open_rate

            # Return the tighter of fixed stoploss or ATR-based
            return max(atr_stop, self.stoploss)

        return self.stoploss
