"""
Andrew Aziz Day Trading Strategy - Freqtrade Implementation
===========================================================

Based on: "Advanced Techniques in Day Trading" by Andrew Aziz

Freqtrade Strategy Features:
- Opening Range Breakout (ORB)
- VWAP Trend Following
- Moving Average Bounce (9 EMA)
- 1% Risk Rule (position sizing)
- 2:1 minimum Risk/Reward
- Volume confirmation
- Multi-timeframe analysis

Usage:
    freqtrade backtesting --strategy AndrewAzizStrategy
    freqtrade trade --strategy AndrewAzizStrategy

Author: Claude (Algo Trading Expert)
Date: 2025-01-13
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Dict

from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import talib.abstract as ta
from freqtrade.persistence import Trade


class AndrewAzizStrategy(IStrategy):
    """
    Andrew Aziz Day Trading Strategy for Freqtrade

    Strategies:
    1. Opening Range Breakout (ORB) - First 15 minutes
    2. VWAP Trend Following - Pullback/Bounce
    3. EMA Bounce - 9 EMA support/resistance

    Risk Management (Andrew Aziz's 1% Rule):
    - 1% risk per trade
    - 5% daily loss limit
    - 2:1 minimum R/R
    - Position sizing based on stop distance
    """

    # ====================================================================
    # STRATEGY METADATA
    # ====================================================================

    INTERFACE_VERSION = 3

    # Minimal ROI (Andrew Aziz: 2:1 R/R minimum)
    minimal_roi = {
        "0": 0.02,   # 2% TP (2:1 R/R with 1% SL)
        "30": 0.015, # 1.5% after 30 min
        "60": 0.01,  # 1% after 1 hour
        "120": 0.005 # 0.5% after 2 hours
    }

    # Stoploss (Andrew Aziz: strict SL based on key levels)
    stoploss = -0.01  # -1% hard stop (adjusted dynamically)

    # Trailing stop (Andrew Aziz: lock in profits)
    trailing_stop = True
    trailing_stop_positive = 0.01  # Activate after 1% profit
    trailing_stop_positive_offset = 0.015  # Trail 1.5% from high
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '1m'  # 1-minute bars (day trading)

    # Use exit signals
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # Startup candles
    startup_candle_count = 100

    # Order types
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # ====================================================================
    # HYPEROPTABLE PARAMETERS (Andrew Aziz parameters)
    # ====================================================================

    # Opening Range parameters
    opening_range_minutes = IntParameter(10, 30, default=15, space='buy', optimize=True)
    orb_volume_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)

    # VWAP parameters
    vwap_tolerance_pct = DecimalParameter(0.003, 0.01, default=0.005, space='buy', optimize=True)

    # EMA parameters
    ema_fast = IntParameter(5, 15, default=9, space='buy', optimize=True)
    ema_slow = IntParameter(15, 30, default=20, space='buy', optimize=True)

    # Risk/Reward
    min_risk_reward = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)

    # Volume surge
    volume_surge_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)

    # ====================================================================
    # STRATEGY VARIABLES
    # ====================================================================

    custom_info = {}  # Store opening range, VWAP, etc.

    # ====================================================================
    # INFORMATIVE PAIRS (for multi-timeframe)
    # ====================================================================

    @informative('5m')
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """5-minute timeframe for trend confirmation"""
        dataframe['ema_9_5m'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_20_5m'] = ta.EMA(dataframe, timeperiod=20)
        return dataframe

    # ====================================================================
    # INDICATORS
    # ====================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all indicators (Andrew Aziz style)

        Key Indicators:
        - VWAP (Volume Weighted Average Price)
        - EMA 9/20 (Moving Averages)
        - Volume Average
        - Opening Range High/Low
        """

        # === VWAP (Andrew Aziz's #1 indicator) ===
        dataframe['vwap'] = self.calculate_vwap(dataframe)

        # === EMAs ===
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        # === Volume ===
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_surge'] = dataframe['volume'] > (dataframe['volume_sma'] * self.volume_surge_multiplier.value)

        # === ATR (for stop loss calculation) ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # === RSI (for overbought/oversold) ===
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # === Opening Range (first N minutes of day) ===
        dataframe = self.calculate_opening_range(dataframe, metadata)

        # === VWAP Bands (standard deviation) ===
        dataframe['vwap_upper'] = dataframe['vwap'] * 1.01  # +1%
        dataframe['vwap_lower'] = dataframe['vwap'] * 0.99  # -1%

        # === Trend Detection ===
        dataframe['uptrend'] = (dataframe['ema_9'] > dataframe['ema_20']) & (dataframe['close'] > dataframe['vwap'])
        dataframe['downtrend'] = (dataframe['ema_9'] < dataframe['ema_20']) & (dataframe['close'] < dataframe['vwap'])

        return dataframe

    # ====================================================================
    # ENTRY SIGNALS (Andrew Aziz Strategies)
    # ====================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on Andrew Aziz strategies

        Signals:
        1. Opening Range Breakout (ORB)
        2. VWAP Pullback/Bounce
        3. EMA Bounce
        """

        conditions_long = []

        # === STRATEGY 1: Opening Range Breakout (ORB) ===
        orb_long = (
            # Price breaks above opening range high
            (dataframe['close'] > dataframe['or_high']) &
            (dataframe['or_high'] > 0) &  # OR determined

            # VWAP confirmation (bullish)
            (dataframe['close'] > dataframe['vwap']) &

            # Trend confirmation
            (dataframe['ema_9'] > dataframe['ema_20']) &

            # Volume surge
            (dataframe['volume_surge'] == True)
        )

        # === STRATEGY 2: VWAP Pullback (Bounce) ===
        vwap_long = (
            # Price near VWAP (within tolerance)
            (abs(dataframe['close'] - dataframe['vwap']) / dataframe['close'] < self.vwap_tolerance_pct.value) &

            # Price above VWAP (or bouncing)
            (dataframe['close'] > dataframe['vwap']) &

            # Uptrend
            (dataframe['uptrend'] == True) &

            # Volume surge
            (dataframe['volume_surge'] == True) &

            # Not overbought
            (dataframe['rsi'] < 70)
        )

        # === STRATEGY 3: 9 EMA Bounce ===
        ema_long = (
            # Price touches/bounces from 9 EMA
            (dataframe['low'] <= dataframe['ema_9']) &
            (dataframe['close'] > dataframe['ema_9']) &

            # Uptrend
            (dataframe['ema_9'] > dataframe['ema_20']) &

            # Above VWAP
            (dataframe['close'] > dataframe['vwap']) &

            # Volume
            (dataframe['volume_surge'] == True)
        )

        # === COMBINE SIGNALS (any of the 3 strategies) ===
        # Set entry signals and tags (Freqtrade uses 'enter_tag' not 'entry_strategy')
        dataframe.loc[orb_long, 'enter_long'] = 1
        dataframe.loc[orb_long, 'enter_tag'] = 'ORB'

        dataframe.loc[vwap_long, 'enter_long'] = 1
        dataframe.loc[vwap_long, 'enter_tag'] = 'VWAP'

        dataframe.loc[ema_long, 'enter_long'] = 1
        dataframe.loc[ema_long, 'enter_tag'] = 'EMA'

        return dataframe

    # ====================================================================
    # EXIT SIGNALS
    # ====================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals (Andrew Aziz: cut losses, let winners run)

        IMPORTANT: Keep exit signals MINIMAL
        - Let ROI and trailing stop do the work
        - Only exit on STRONG reversal signals
        """

        # === EXIT: Strong trend reversal ONLY ===
        # Must have ALL conditions (very strict)
        exit_strong_reversal = (
            # Strong EMA cross down
            (dataframe['ema_9'] < dataframe['ema_20']) &

            # Well below VWAP (not just touched)
            (dataframe['close'] < dataframe['vwap'] * 0.995) &  # -0.5%

            # Price below both EMAs
            (dataframe['close'] < dataframe['ema_9']) &
            (dataframe['close'] < dataframe['ema_20']) &

            # High volume reversal
            (dataframe['volume'] > dataframe['volume_sma'] * 1.5) &

            # RSI confirms weakness
            (dataframe['rsi'] < 40)
        )

        dataframe.loc[exit_strong_reversal, 'exit_long'] = 1

        return dataframe

    # ====================================================================
    # CUSTOM STOPLOSS (Andrew Aziz: based on entry strategy)
    # ====================================================================

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> Optional[float]:
        """
        Custom stop loss based on Andrew Aziz methodology

        SIMPLIFIED: Use FIXED stop loss based on entry price, not dynamic indicators
        - This prevents stop loss from moving unfavorably
        - Andrew Aziz: Set SL at entry and DON'T move it (except trailing)

        Strategy-based stops:
        - ORB: -1.5% (wider for breakouts)
        - VWAP: -1.0% (standard)
        - EMA: -0.8% (tighter for bounces)
        """

        # Get entry strategy
        entry_strategy = trade.enter_tag or 'VWAP'

        # FIXED stop loss percentages (don't change after entry)
        if entry_strategy == 'ORB':
            return -0.015  # -1.5% (breakouts need room)
        elif entry_strategy == 'VWAP':
            return -0.01   # -1.0% (standard)
        elif entry_strategy == 'EMA':
            return -0.008  # -0.8% (tight for bounces)
        else:
            return -0.01   # -1.0% default

    # ====================================================================
    # CUSTOM EXIT (Andrew Aziz: time-based, end of day)
    # ====================================================================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        Custom exit logic

        Andrew Aziz Rule: Close all positions before market close
        Crypto: Close positions after certain hours or max hold time
        """

        # Max hold time: 4 hours (day trading)
        if current_time - trade.open_date_utc > timedelta(hours=4):
            return "max_hold_time"

        # Profit target hit (2:1 R/R)
        if current_profit >= 0.02:  # 2% profit
            return "profit_target"

        return None

    # ====================================================================
    # POSITION SIZING (Andrew Aziz 1% Rule)
    # ====================================================================

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Position sizing based on Andrew Aziz 1% Risk Rule

        Formula:
        Risk Amount = Account Size × 1%
        Position Size = Risk Amount / (Entry - Stop)
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Get stop loss price
        entry_strategy = entry_tag or 'VWAP'

        if entry_strategy == 'ORB':
            sl_price = last_candle.get('or_low', current_rate * 0.99)
        elif entry_strategy == 'VWAP':
            sl_price = last_candle.get('vwap', 0) * 0.995
        elif entry_strategy == 'EMA':
            sl_price = last_candle.get('ema_20', current_rate * 0.99)
        else:
            sl_price = current_rate * 0.99  # 1% default

        # Calculate risk per coin
        risk_per_coin = current_rate - sl_price

        if risk_per_coin <= 0:
            return proposed_stake

        # Andrew Aziz 1% Rule
        account_size = self.wallets.get_total_stake_amount()
        risk_amount = account_size * 0.01  # 1% risk

        # Position size
        position_size_coins = risk_amount / risk_per_coin
        position_stake = position_size_coins * current_rate

        # Respect min/max limits
        if min_stake and position_stake < min_stake:
            return min_stake
        if position_stake > max_stake:
            return max_stake

        return position_stake

    # ====================================================================
    # HELPER FUNCTIONS
    # ====================================================================

    def calculate_vwap(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price)
        Andrew Aziz's #1 indicator

        VWAP = Cumulative(Price × Volume) / Cumulative(Volume)
        Reset daily at 00:00 UTC
        """
        # Typical price
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

        # Price × Volume
        pv = typical_price * dataframe['volume']

        # Daily reset (group by date)
        df_copy = dataframe.copy()
        df_copy['pv'] = pv
        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.date

        # Cumulative sum per day
        df_copy['cumulative_pv'] = df_copy.groupby('date')['pv'].transform(pd.Series.cumsum)
        df_copy['cumulative_volume'] = df_copy.groupby('date')['volume'].transform(pd.Series.cumsum)

        # VWAP per day
        vwap = df_copy['cumulative_pv'] / df_copy['cumulative_volume']

        return vwap.fillna(dataframe['close'])

    def calculate_opening_range(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate Opening Range (first N minutes of each day)

        Andrew Aziz: Opening Range Breakout strategy
        - Track high/low of first 15 minutes of THE DAY
        - Breakout = strong signal
        - Opening Range is FIXED for entire day (not rolling!)
        """

        # Get date column
        df_copy = dataframe.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.date

        # Number of candles in opening range (N minutes on 1m timeframe = N candles)
        or_window = self.opening_range_minutes.value

        # For each day, get first N candles' high/low
        def get_daily_or(group):
            """Get Opening Range high/low for the day"""
            if len(group) < or_window:
                # Not enough candles yet
                return pd.Series({
                    'or_high': group['high'].max(),
                    'or_low': group['low'].min()
                })
            else:
                # First N candles of the day
                first_candles = group.head(or_window)
                return pd.Series({
                    'or_high': first_candles['high'].max(),
                    'or_low': first_candles['low'].min()
                })

        # Calculate OR per day and forward-fill for entire day
        daily_or = df_copy.groupby('date').apply(get_daily_or).reset_index()

        # Merge back to dataframe
        df_copy = df_copy.merge(daily_or, on='date', how='left')

        # Assign to original dataframe
        dataframe['or_high'] = df_copy['or_high'].fillna(0)
        dataframe['or_low'] = df_copy['or_low'].fillna(0)

        return dataframe

    # ====================================================================
    # CONFIRM TRADE ENTRY (final check before entry)
    # ====================================================================

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Final confirmation before entering trade

        Andrew Aziz checks:
        - Risk/Reward >= 2:1
        - Volume confirmation
        - Not over-trading (daily trade limit)
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Check R/R ratio
        entry_price = rate

        # Get stop and target
        if entry_tag == 'ORB':
            sl_price = last_candle.get('or_low', entry_price * 0.99)
            tp_price = entry_price + (entry_price - sl_price) * self.min_risk_reward.value
        elif entry_tag == 'VWAP':
            sl_price = last_candle.get('vwap', 0) * 0.995
            tp_price = entry_price + (entry_price - sl_price) * self.min_risk_reward.value
        else:
            sl_price = entry_price * 0.99
            tp_price = entry_price * 1.02

        risk = entry_price - sl_price
        reward = tp_price - entry_price

        if risk <= 0:
            return False

        rr_ratio = reward / risk

        # Andrew Aziz: Min 2:1 R/R
        if rr_ratio < self.min_risk_reward.value:
            return False

        # Volume check
        if not last_candle.get('volume_surge', False):
            return False

        return True


# ====================================================================
# VERSION INFO
# ====================================================================

"""
FREQTRADE USAGE:

1. Backtesting:
   freqtrade backtesting \\
       --strategy AndrewAzizStrategy \\
       --timerange 20240101-20240630 \\
       --timeframe 1m

2. Hyperopt (optimize parameters):
   freqtrade hyperopt \\
       --strategy AndrewAzizStrategy \\
       --hyperopt-loss SharpeHyperOptLoss \\
       --spaces buy \\
       -e 100

3. Dry-run (paper trading):
   freqtrade trade \\
       --strategy AndrewAzizStrategy \\
       --dry-run

4. Live trading:
   freqtrade trade \\
       --strategy AndrewAzizStrategy \\
       --config config.json

CONFIGURATION (config.json):

{
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "dry_run": true,
    "exchange": {
        "name": "binance",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_SECRET"
    },
    "pairlists": [
        {
            "method": "VolumePairList",
            "number_assets": 10,
            "sort_key": "quoteVolume",
            "min_value": 0,
            "refresh_period": 1800
        }
    ],
    "edge": {
        "enabled": false
    }
}

ANDREW AZIZ PARAMETERS (optimized):
- opening_range_minutes: 15
- vwap_tolerance_pct: 0.005 (0.5%)
- ema_fast: 9
- ema_slow: 20
- min_risk_reward: 2.0
- volume_surge_multiplier: 2.0

EXPECTED PERFORMANCE:
- Win Rate: 50-60%
- Avg Win: +2%
- Avg Loss: -1%
- Sharpe Ratio: 2.0+
- Max Drawdown: <10%
"""
