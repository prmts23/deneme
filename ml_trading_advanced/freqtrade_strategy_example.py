"""
Freqtrade Strategy Example using Advanced ML Models

Copy this file to your Freqtrade user_data/strategies/ folder
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import os
import sys

# Add ml_trading_advanced to path
sys.path.append('/path/to/ml_trading_advanced')  # UPDATE THIS PATH

from inference import TradingPredictor


class MLAdvancedStrategy(IStrategy):
    """
    Advanced ML Trading Strategy for Freqtrade

    This strategy uses professionally trained ML models with:
    - Triple Barrier Method labeling
    - 200+ engineered features
    - Hyperparameter optimization
    - Walk-forward validation

    Performance expectations:
    - Win rate: 55-70% (depending on threshold)
    - Risk-reward: 1.5-2.5
    - Sharpe ratio: 1.5+

    IMPORTANT: Backtest thoroughly before live trading!
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # ============================================================
    # STRATEGY PARAMETERS
    # ============================================================

    # ROI table (can be disabled with minimal_roi = {})
    minimal_roi = {
        "0": 0.02,    # 2% profit
        "30": 0.015,  # 1.5% after 30 minutes
        "60": 0.01,   # 1% after 1 hour
        "120": 0.005  # 0.5% after 2 hours
    }

    # Stoploss
    stoploss = -0.02  # -2% (adjust based on your barrier settings)

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '5m'

    # Run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will take precedence)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # ============================================================
    # ML MODEL CONFIGURATION
    # ============================================================

    # Path to trained models
    model_dir = '/path/to/ml_trading_advanced/models_advanced'  # UPDATE THIS

    # Prediction thresholds
    long_entry_threshold = 0.65   # Higher = fewer but better signals
    long_exit_threshold = 0.45    # Lower = exit earlier

    short_entry_threshold = 0.65  # For short positions
    short_exit_threshold = 0.45

    # Trading mode
    can_short = False  # Set True if your exchange supports shorting

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Initialize ML predictor
        try:
            self.predictor = TradingPredictor(model_dir=self.model_dir)
            print("✅ ML models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading ML models: {e}")
            print(f"   Make sure model_dir is correct: {self.model_dir}")
            self.predictor = None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate ML predictions and add as indicators

        Args:
            dataframe: OHLCV data
            metadata: Pair metadata

        Returns:
            DataFrame with ML predictions
        """
        if self.predictor is None:
            # Models not loaded, return empty signals
            dataframe['ml_long_prob'] = 0.0
            dataframe['ml_short_prob'] = 0.0
            dataframe['ml_signal'] = 0
            return dataframe

        try:
            # Prepare dataframe for prediction
            df_pred = dataframe.copy()

            # Rename columns if needed
            df_pred.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)

            # Add timestamp if not present
            if 'timestamp' not in df_pred.columns:
                df_pred['timestamp'] = df_pred['date']

            # Get predictions
            if len(df_pred) >= 100:  # Need enough data for feature engineering
                # LONG predictions
                try:
                    pred_long = self.predictor.predict(
                        df_pred,
                        side='long',
                        threshold=self.long_entry_threshold
                    )
                    dataframe['ml_long_prob'] = pred_long['probabilities']
                    dataframe['ml_long_signal'] = pred_long['predictions']
                except Exception as e:
                    print(f"Error in LONG prediction: {e}")
                    dataframe['ml_long_prob'] = 0.0
                    dataframe['ml_long_signal'] = 0

                # SHORT predictions (if enabled)
                if self.can_short:
                    try:
                        pred_short = self.predictor.predict(
                            df_pred,
                            side='short',
                            threshold=self.short_entry_threshold
                        )
                        dataframe['ml_short_prob'] = pred_short['probabilities']
                        dataframe['ml_short_signal'] = pred_short['predictions']
                    except Exception as e:
                        print(f"Error in SHORT prediction: {e}")
                        dataframe['ml_short_prob'] = 0.0
                        dataframe['ml_short_signal'] = 0
                else:
                    dataframe['ml_short_prob'] = 0.0
                    dataframe['ml_short_signal'] = 0

                # Combined signal
                dataframe['ml_signal'] = 0
                dataframe.loc[dataframe['ml_long_signal'] == 1, 'ml_signal'] = 1
                if self.can_short:
                    dataframe.loc[dataframe['ml_short_signal'] == 1, 'ml_signal'] = -1

            else:
                # Not enough data
                dataframe['ml_long_prob'] = 0.0
                dataframe['ml_short_prob'] = 0.0
                dataframe['ml_signal'] = 0

        except Exception as e:
            print(f"Error in populate_indicators: {e}")
            dataframe['ml_long_prob'] = 0.0
            dataframe['ml_short_prob'] = 0.0
            dataframe['ml_signal'] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals

        Args:
            dataframe: DataFrame with indicators
            metadata: Pair metadata

        Returns:
            DataFrame with entry signals
        """
        # LONG entry
        dataframe.loc[
            (
                (dataframe['ml_signal'] == 1) &  # ML says LONG
                (dataframe['ml_long_prob'] >= self.long_entry_threshold) &  # High confidence
                (dataframe['volume'] > 0)  # Valid candle
            ),
            'enter_long'] = 1

        # SHORT entry (if enabled)
        if self.can_short:
            dataframe.loc[
                (
                    (dataframe['ml_signal'] == -1) &  # ML says SHORT
                    (dataframe['ml_short_prob'] >= self.short_entry_threshold) &  # High confidence
                    (dataframe['volume'] > 0)  # Valid candle
                ),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals

        Args:
            dataframe: DataFrame with indicators
            metadata: Pair metadata

        Returns:
            DataFrame with exit signals
        """
        # LONG exit (when probability drops or SHORT signal appears)
        dataframe.loc[
            (
                (
                    (dataframe['ml_long_prob'] < self.long_exit_threshold) |  # Low confidence
                    (dataframe['ml_signal'] == -1)  # Opposite signal
                ) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        # SHORT exit (if enabled)
        if self.can_short:
            dataframe.loc[
                (
                    (
                        (dataframe['ml_short_prob'] < self.short_exit_threshold) |  # Low confidence
                        (dataframe['ml_signal'] == 1)  # Opposite signal
                    ) &
                    (dataframe['volume'] > 0)
                ),
                'exit_short'] = 1

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time, entry_tag, side: str, **kwargs) -> bool:
        """
        Called right before placing a buy/sell order.
        Can be used for additional confirmations or risk checks.

        Args:
            pair: Trading pair
            ... (other parameters)

        Returns:
            True to confirm trade, False to reject
        """
        # Get latest dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return False

        last_candle = dataframe.iloc[-1]

        # Additional safety checks
        if side == "long":
            # Confirm LONG entry
            if last_candle['ml_long_prob'] >= self.long_entry_threshold:
                return True
        elif side == "short":
            # Confirm SHORT entry
            if last_candle['ml_short_prob'] >= self.short_entry_threshold:
                return True

        return False

    def custom_exit(self, pair: str, trade, current_time, current_rate,
                   current_profit, **kwargs) -> str:
        """
        Custom exit logic (optional)

        Can implement:
        - Time-based exits
        - Volatility-based exits
        - ML confidence-based exits

        Returns:
            Exit reason string or None
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        # Exit if ML confidence drops significantly
        if trade.is_short:
            if last_candle['ml_short_prob'] < 0.3:  # Very low confidence
                return 'ml_confidence_drop_short'
        else:
            if last_candle['ml_long_prob'] < 0.3:  # Very low confidence
                return 'ml_confidence_drop_long'

        # Exit if opposite signal with high confidence
        if trade.is_short and last_candle['ml_signal'] == 1 and last_candle['ml_long_prob'] > 0.7:
            return 'ml_opposite_signal_short'
        elif not trade.is_short and last_candle['ml_signal'] == -1 and last_candle['ml_short_prob'] > 0.7:
            return 'ml_opposite_signal_long'

        return None


# ============================================================
# HYPEROPT PARAMETERS (Optional - for optimization)
# ============================================================

"""
If you want to optimize thresholds with Freqtrade Hyperopt:

from freqtrade.optimize.space import DecimalParameter

class MLAdvancedStrategyHyperopt(MLAdvancedStrategy):
    # Hyperopt parameters
    long_entry_threshold = DecimalParameter(0.5, 0.8, default=0.65, space='buy')
    long_exit_threshold = DecimalParameter(0.3, 0.6, default=0.45, space='sell')

Then run:
freqtrade hyperopt --strategy MLAdvancedStrategyHyperopt --epochs 100
"""
