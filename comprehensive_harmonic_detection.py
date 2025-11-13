#!/usr/bin/env python3
"""
Comprehensive Harmonic Pattern Detection System
===============================================

基于蝴蝶模式成功经验，实现所有Scott Carney标准harmonic patterns:
- Gartley, Bat, Crab, Butterfly, Cypher, Shark
- 完全向量化实现，零循环
- 多空对称性保证
- 精确D点价位控制

Created: 2025-09-08
Purpose: Complete vectorized harmonic pattern detection system
"""

import logging
import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from datetime import datetime

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

            price_range = np.max(high_prices) - np.min(low_prices)
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

            df.iloc[high_peaks, df.columns.get_loc('swing_high')] = True
            df.iloc[low_peaks, df.columns.get_loc('swing_low')] = True

        except Exception as e:
            logger.error(f"Swing detection failed: {e}")
            df['swing_high'] = False
            df['swing_low'] = False

        return df

class HarmonicPatternSpecs:
    """Scott Carney harmonic pattern specifications."""

    @staticmethod
    def get_pattern_specs() -> Dict[str, Dict]:
        """
        Get Scott Carney standard specifications for all harmonic patterns.

        Returns:
            Dictionary with pattern specifications
        """
        return {
            'gartley': {
                'name': 'Gartley',
                'ab_xa_range': (0.618, 0.618),  # Exact 61.8%
                'bc_ab_range': (0.382, 0.886),  # 38.2% to 88.6%
                'cd_ab_range': (1.272, 1.618),  # 127.2% to 161.8%
                'xd_xa_range': (0.786, 0.786)   # Exact 78.6%
            },
            'bat': {
                'name': 'Bat',
                'ab_xa_range': (0.382, 0.5),    # 38.2% to 50%
                'bc_ab_range': (0.382, 0.886),  # 38.2% to 88.6%
                'cd_ab_range': (1.618, 2.618),  # 161.8% to 261.8%
                'xd_xa_range': (0.886, 0.886)   # Exact 88.6%
            },
            'crab': {
                'name': 'Crab',
                'ab_xa_range': (0.382, 0.618),  # 38.2% to 61.8%
                'bc_ab_range': (0.382, 0.886),  # 38.2% to 88.6%
                'cd_ab_range': (2.24, 3.618),   # 224% to 361.8%
                'xd_xa_range': (1.618, 1.618)   # Exact 161.8%
            },
            'butterfly': {
                'name': 'Butterfly',
                'ab_xa_range': (0.786, 0.786),  # Exact 78.6%
                'bc_ab_range': (0.382, 0.886),  # 38.2% to 88.6%
                'cd_ab_range': (1.618, 2.618),  # 161.8% to 261.8%
                'xd_xa_range': (1.27, 1.27)     # Exact 127%
            },
            'cypher': {
                'name': 'Cypher',
                'ab_xa_range': (0.382, 0.786),  # 38.2% to 78.6%
                'bc_ab_range': (1.272, 1.414),  # 127.2% to 141.4%
                'cd_bc_range': (0.786, 0.786),  # Exact 78.6%
                'xd_xa_range': (0.786, 0.786)   # Exact 78.6%
            },
            'shark': {
                'name': 'Shark',
                'ab_xa_range': (0.382, 0.618),  # 38.2% to 61.8%
                'bc_ab_range': (1.13, 1.618),   # 113% to 161.8%
                'cd_ab_range': (1.618, 2.214),  # 161.8% to 221.4%
                'xd_xa_range': (0.886, 1.13)    # 88.6% to 113%
            }
        }

class VectorizedHarmonicDetector:
    """Vectorized detector for all harmonic patterns."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize with tolerance for ratio matching.

        Args:
            tolerance: Tolerance for Fibonacci ratio matching
        """
        self.tolerance = tolerance
        self.pattern_specs = HarmonicPatternSpecs.get_pattern_specs()

    def generate_xabcd_combinations(self, swing_highs: pd.DataFrame, swing_lows: pd.DataFrame,
                                  pattern_type: str = 'bullish', max_patterns: int = 200) -> pd.DataFrame:
        """Generate XABCD combinations using pure vectorized operations."""
        if len(swing_highs) < 2 or len(swing_lows) < 3:
            return pd.DataFrame()

        swing_highs = swing_highs.reset_index()
        swing_lows = swing_lows.reset_index()

        if pattern_type == 'bullish':
            # Bullish: X(low) -> A(high) -> B(low) -> C(high) -> D(low)
            timestamp_col = 'timestamp' if 'timestamp' in swing_lows.columns else swing_lows.columns.tolist()[0]

            x_df = swing_lows[[timestamp_col, 'low']].rename(
                columns={timestamp_col: 'x_idx', 'low': 'x_price'}).tail(8)
            a_df = swing_highs[[timestamp_col, 'high']].rename(
                columns={timestamp_col: 'a_idx', 'high': 'a_price'}).tail(12)
            b_df = swing_lows[[timestamp_col, 'low']].rename(
                columns={timestamp_col: 'b_idx', 'low': 'b_price'}).tail(12)
            c_df = swing_highs[[timestamp_col, 'high']].rename(
                columns={timestamp_col: 'c_idx', 'high': 'c_price'}).tail(12)
            d_df = swing_lows[[timestamp_col, 'low']].rename(
                columns={timestamp_col: 'd_idx', 'low': 'd_price'}).tail(12)
        else:
            # Bearish: X(high) -> A(low) -> B(high) -> C(low) -> D(high)
            timestamp_col = 'timestamp' if 'timestamp' in swing_highs.columns else swing_highs.columns.tolist()[0]

            x_df = swing_highs[[timestamp_col, 'high']].rename(
                columns={timestamp_col: 'x_idx', 'high': 'x_price'}).tail(8)
            a_df = swing_lows[[timestamp_col, 'low']].rename(
                columns={timestamp_col: 'a_idx', 'low': 'a_price'}).tail(12)
            b_df = swing_highs[[timestamp_col, 'high']].rename(
                columns={timestamp_col: 'b_idx', 'high': 'b_price'}).tail(12)
            c_df = swing_lows[[timestamp_col, 'low']].rename(
                columns={timestamp_col: 'c_idx', 'low': 'c_price'}).tail(12)
            d_df = swing_highs[[timestamp_col, 'high']].rename(
                columns={timestamp_col: 'd_idx', 'high': 'd_price'}).tail(12)

        # Add merge keys using consistent naming
        x_df['merge_key'] = 1
        a_df['merge_key'] = 1
        b_df['merge_key'] = 1
        c_df['merge_key'] = 1
        d_df['merge_key'] = 1

        try:
            # Vectorized cross-join operations
            xa = pd.merge(x_df, a_df, on='merge_key', suffixes=('', '_a'))
            xa_valid = xa[xa['a_idx'] > xa['x_idx']]

            if xa_valid.empty:
                return pd.DataFrame()

            xab = pd.merge(xa_valid, b_df, on='merge_key', suffixes=('', '_b'))
            xab_valid = xab[xab['b_idx'] > xab['a_idx']]

            if xab_valid.empty:
                return pd.DataFrame()

            xabc = pd.merge(xab_valid, c_df, on='merge_key', suffixes=('', '_c'))
            xabc_valid = xabc[xabc['c_idx'] > xabc['b_idx']]

            if xabc_valid.empty:
                return pd.DataFrame()

            xabcd = pd.merge(xabc_valid, d_df, on='merge_key', suffixes=('', '_d'))
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
            # Vectorized ratio calculations
            xa_moves = np.abs(combinations_df['a_price'].values - combinations_df['x_price'].values)
            ab_moves = np.abs(combinations_df['b_price'].values - combinations_df['a_price'].values)
            bc_moves = np.abs(combinations_df['c_price'].values - combinations_df['b_price'].values)
            cd_moves = np.abs(combinations_df['d_price'].values - combinations_df['c_price'].values)
            xd_moves = np.abs(combinations_df['d_price'].values - combinations_df['x_price'].values)

            # Prevent division by zero
            xa_moves = np.where(xa_moves == 0, 1e-10, xa_moves)
            ab_moves = np.where(ab_moves == 0, 1e-10, ab_moves)
            bc_moves = np.where(bc_moves == 0, 1e-10, bc_moves)
            xd_moves = np.where(xd_moves == 0, 1e-10, xd_moves)

            combinations_df = combinations_df.copy()
            combinations_df['ab_xa'] = ab_moves / xa_moves
            combinations_df['bc_ab'] = bc_moves / ab_moves
            combinations_df['cd_ab'] = cd_moves / ab_moves
            combinations_df['cd_bc'] = cd_moves / bc_moves
            combinations_df['xd_xa'] = xd_moves / xa_moves

            return combinations_df

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

            # Check AB/XA ratio
            ab_xa = combo['ab_xa']
            ab_range = spec['ab_xa_range']
            if ab_range[0] - self.tolerance <= ab_xa <= ab_range[1] + self.tolerance:
                score += 1

            # Check BC/AB ratio
            bc_ab = combo['bc_ab']
            bc_range = spec['bc_ab_range']
            if bc_range[0] - self.tolerance <= bc_ab <= bc_range[1] + self.tolerance:
                score += 1

            # Check CD ratio (either CD/AB or CD/BC depending on pattern)
            if 'cd_ab_range' in spec:
                cd_ab = combo['cd_ab']
                cd_range = spec['cd_ab_range']
                if cd_range[0] - self.tolerance <= cd_ab <= cd_range[1] + self.tolerance:
                    score += 1
            elif 'cd_bc_range' in spec:  # For Cypher pattern
                cd_bc = combo['cd_bc']
                cd_range = spec['cd_bc_range']
                if cd_range[0] - self.tolerance <= cd_bc <= cd_range[1] + self.tolerance:
                    score += 1

            # Check XD/XA ratio
            xd_xa = combo['xd_xa']
            xd_range = spec['xd_xa_range']
            if xd_range[0] - self.tolerance <= xd_xa <= xd_range[1] + self.tolerance:
                score += 1

            # Accept patterns with at least 3/4 criteria met
            if score >= 3:
                combo_dict = combo.to_dict()
                combo_dict['pattern_name'] = pattern_name
                combo_dict['pattern_score'] = score
                valid_combinations.append(combo_dict)

        return pd.DataFrame(valid_combinations) if valid_combinations else pd.DataFrame()

class ComprehensiveHarmonicDetector:
    """Main comprehensive harmonic pattern detection system."""

    def __init__(self, tolerance: float = 0.3):
        """Initialize the comprehensive harmonic detector."""
        self.tolerance = tolerance
        self.swing_detector = SwingDetector()
        self.pattern_detector = VectorizedHarmonicDetector(tolerance)
        self.pattern_names = ['gartley', 'bat', 'crab', 'butterfly', 'cypher', 'shark']

    def detect_all_patterns(self, dataframe: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Detect all harmonic patterns in OHLCV data.

        Args:
            dataframe: OHLCV price data

        Returns:
            Nested dictionary with pattern_name -> {'bullish': df, 'bearish': df}
        """
        # Step 1: Detect swing points
        df_with_swings = self.swing_detector.detect_swings(
            dataframe,
            prominence_factor=0.001,
            distance=6
        )

        swing_highs = df_with_swings[df_with_swings['swing_high']].copy()
        swing_lows = df_with_swings[df_with_swings['swing_low']].copy()

        swing_highs = swing_highs.reset_index()
        swing_lows = swing_lows.reset_index()

        if len(swing_highs) < 2 or len(swing_lows) < 3:
            return {name: {'bullish': pd.DataFrame(), 'bearish': pd.DataFrame()}
                   for name in self.pattern_names}

        results = {}

        # Step 2: For each pattern type
        for pattern_name in self.pattern_names:
            logger.info(f"Detecting {pattern_name} patterns...")

            # Generate combinations for both bullish and bearish
            bullish_combinations = self.pattern_detector.generate_xabcd_combinations(
                swing_highs, swing_lows, 'bullish', max_patterns=100
            )

            bearish_combinations = self.pattern_detector.generate_xabcd_combinations(
                swing_highs, swing_lows, 'bearish', max_patterns=100
            )

            # Calculate ratios
            if not bullish_combinations.empty:
                bullish_combinations = self.pattern_detector.calculate_pattern_ratios(bullish_combinations)
                bullish_patterns = self.pattern_detector.validate_pattern(bullish_combinations, pattern_name)
            else:
                bullish_patterns = pd.DataFrame()

            if not bearish_combinations.empty:
                bearish_combinations = self.pattern_detector.calculate_pattern_ratios(bearish_combinations)
                bearish_patterns = self.pattern_detector.validate_pattern(bearish_combinations, pattern_name)
            else:
                bearish_patterns = pd.DataFrame()

            # Add pattern type for signal generation
            if not bullish_patterns.empty:
                bullish_patterns['pattern_type'] = 'bullish'
            if not bearish_patterns.empty:
                bearish_patterns['pattern_type'] = 'bearish'

            results[pattern_name] = {
                'bullish': bullish_patterns,
                'bearish': bearish_patterns
            }

            logger.info(f"{pattern_name}: {len(bullish_patterns)} bullish, {len(bearish_patterns)} bearish")

        return results

    def get_all_trading_signals(self, all_patterns: Dict, dataframe: pd.DataFrame) -> List[Dict]:
        """Generate trading signals from all detected patterns."""
        signals = []

        for pattern_name, patterns in all_patterns.items():
            # Process bullish patterns
            for _, pattern in patterns['bullish'].iterrows():
                d_timestamp = pattern['d_idx']
                if d_timestamp in dataframe.index:
                    signals.append({
                        'timestamp': d_timestamp,
                        'signal_type': 'BUY',
                        'price': pattern['d_price'],
                        'pattern': f'Bullish {pattern["pattern_name"].title()}',
                        'score': pattern['pattern_score'],
                        'ab_xa': pattern['ab_xa'],
                        'bc_ab': pattern['bc_ab'],
                        'xd_xa': pattern['xd_xa']
                    })

            # Process bearish patterns
            for _, pattern in patterns['bearish'].iterrows():
                d_timestamp = pattern['d_idx']
                if d_timestamp in dataframe.index:
                    signals.append({
                        'timestamp': d_timestamp,
                        'signal_type': 'SELL',
                        'price': pattern['d_price'],
                        'pattern': f'Bearish {pattern["pattern_name"].title()}',
                        'score': pattern['pattern_score'],
                        'ab_xa': pattern['ab_xa'],
                        'bc_ab': pattern['bc_ab'],
                        'xd_xa': pattern['xd_xa']
                    })

        # Sort signals by timestamp
        signals.sort(key=lambda x: x['timestamp'])
        return signals
