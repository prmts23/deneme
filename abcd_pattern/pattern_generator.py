"""
ABCD Harmonic Pattern Generator
Generates synthetic ABCD patterns with various timeframes and noise levels
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta
import logging

from .config import PatternConfig

logger = logging.getLogger(__name__)


class ABCDPatternGenerator:
    """Generator for ABCD harmonic patterns"""

    def __init__(self, config: Optional[PatternConfig] = None):
        """
        Initialize pattern generator

        Args:
            config: Pattern configuration
        """
        self.config = config or PatternConfig()

    def generate_pattern(
        self,
        total_bars: int,
        bullish: bool = True,
        base_price: float = 100.0
    ) -> np.ndarray:
        """
        Generate ABCD pattern prices

        Args:
            total_bars: Total number of bars in pattern
            bullish: True for bullish (bottom) pattern, False for bearish (top)
            base_price: Starting price level

        Returns:
            Array of prices for the pattern
        """
        if total_bars < 4:
            raise ValueError(f"ABCD requires at least 4 bars, got {total_bars}")

        # Direction multiplier
        sign = -1 if bullish else 1

        # Generate pattern points
        AB_len = sign * np.random.uniform(
            self.config.min_ab_size,
            self.config.max_ab_size
        )

        retrace = np.random.uniform(
            self.config.min_retracement,
            self.config.max_retracement
        )

        abcd_ratio = np.random.uniform(
            self.config.min_abcd_ratio,
            self.config.max_abcd_ratio
        )

        # Calculate points (normalized)
        A = 0.0
        B = A + AB_len
        C = B + retrace * (A - B)
        CD_len = abcd_ratio * (B - A)
        D = C + CD_len

        # Distribute bars across legs
        remaining = total_bars - 1
        n_AB = max(1, np.random.randint(1, remaining - 1))
        n_BC = max(1, np.random.randint(1, remaining - n_AB))
        n_CD = remaining - n_AB - n_BC

        # Generate interpolated prices for each leg
        leg_AB = np.linspace(A, B, n_AB, endpoint=False)
        leg_BC = np.linspace(B, C, n_BC, endpoint=False)
        leg_CD = np.linspace(C, D, n_CD, endpoint=False)

        # Combine legs
        prices = np.concatenate([leg_AB, leg_BC, leg_CD, np.array([D])])

        # Add noise
        noise = np.random.normal(0, self.config.noise_level, size=prices.shape)
        prices_noisy = prices + noise

        # Scale to base price
        prices_scaled = base_price * (1 + prices_noisy / 10)

        return prices_scaled

    def generate_random_prices(
        self,
        total_bars: int,
        base_price: float = 100.0,
        volatility: float = 0.02
    ) -> np.ndarray:
        """
        Generate random walk prices (no pattern)

        Args:
            total_bars: Number of bars
            base_price: Starting price
            volatility: Price volatility

        Returns:
            Array of random prices
        """
        returns = np.random.normal(0, volatility, total_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        return prices

    def generate_ohlcv(
        self,
        prices: np.ndarray,
        start_time: Optional[datetime] = None,
        timeframe: str = "1h",
        base_volume: float = 1000.0
    ) -> pd.DataFrame:
        """
        Convert prices to OHLCV DataFrame

        Args:
            prices: Array of prices
            start_time: Starting timestamp
            timeframe: Candle timeframe
            base_volume: Base volume for candles

        Returns:
            OHLCV DataFrame with datetime index
        """
        n_bars = len(prices)

        if start_time is None:
            start_time = datetime.now()

        # Generate timestamps
        timeframe_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1)
        }
        delta = timeframe_map.get(timeframe, timedelta(hours=1))
        timestamps = [start_time + i * delta for i in range(n_bars)]

        # Generate OHLC from prices
        ohlc_data = []
        for i, price in enumerate(prices):
            # Add some intra-bar movement
            high_offset = abs(np.random.normal(0, 0.002))
            low_offset = abs(np.random.normal(0, 0.002))

            if i == 0:
                open_price = price
            else:
                # Open at previous close with small gap
                open_price = ohlc_data[-1]["Close"] * (1 + np.random.normal(0, 0.001))

            close_price = price

            # Ensure high/low logic
            high = max(open_price, close_price) * (1 + high_offset)
            low = min(open_price, close_price) * (1 - low_offset)

            # Generate volume
            volume = base_volume * np.random.uniform(0.5, 1.5)

            ohlc_data.append({
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close_price,
                "Volume": volume
            })

        df = pd.DataFrame(ohlc_data, index=timestamps)
        df.index.name = "Date"

        return df

    def generate_pattern_ohlcv(
        self,
        total_bars: int,
        bullish: bool = True,
        with_pattern: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate complete OHLCV DataFrame with or without pattern

        Args:
            total_bars: Number of bars
            bullish: Bullish or bearish pattern
            with_pattern: Include pattern or generate random
            **kwargs: Additional arguments for generate_ohlcv

        Returns:
            OHLCV DataFrame
        """
        base_price = kwargs.pop("base_price", np.random.uniform(50, 150))

        if with_pattern:
            prices = self.generate_pattern(total_bars, bullish, base_price)
        else:
            prices = self.generate_random_prices(total_bars, base_price)

        df = self.generate_ohlcv(prices, **kwargs)
        return df

    def validate_pattern(
        self,
        A: float, B: float, C: float, D: float,
        tolerance: float = 0.1
    ) -> Tuple[bool, str]:
        """
        Validate if points form a valid ABCD pattern

        Args:
            A, B, C, D: Pattern points
            tolerance: Tolerance for ratio validation

        Returns:
            (is_valid, message)
        """
        # Calculate retracement BC/AB
        AB = abs(B - A)
        BC = abs(C - B)

        if AB == 0:
            return False, "AB length is zero"

        retracement = BC / AB

        # Check if retracement is in valid range
        if not (self.config.min_retracement - tolerance <= retracement <=
                self.config.max_retracement + tolerance):
            return False, f"Retracement {retracement:.3f} out of valid range"

        # Calculate ABCD ratio
        CD = abs(D - C)
        abcd_ratio = CD / AB

        if not (self.config.min_abcd_ratio - tolerance <= abcd_ratio <=
                self.config.max_abcd_ratio + tolerance):
            return False, f"ABCD ratio {abcd_ratio:.3f} out of valid range"

        return True, "Valid ABCD pattern"
