"""
Advanced Feature Engineering for ML Trading
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis
from ta import add_all_ta_features
from ta.utils import dropna
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for trading ML models
    """

    def __init__(self, config):
        self.config = config

    def engineer_all_features(self, df):
        """Main method to create all features"""
        print("🔧 Engineering advanced features...\n")

        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 1. Time features
        if self.config.USE_TIME_FEATURES:
            df = self._add_time_features(df)
            print("   ✓ Time features added")

        # 2. Price action features
        df = self._add_price_action_features(df)
        print("   ✓ Price action features added")

        # 3. Volume features
        if self.config.USE_VOLUME_FEATURES:
            df = self._add_volume_features(df)
            print("   ✓ Volume features added")

        # 4. Volatility features
        if self.config.USE_VOLATILITY_FEATURES:
            df = self._add_volatility_features(df)
            print("   ✓ Volatility features added")

        # 5. Market regime features
        if self.config.USE_REGIME_FEATURES:
            df = self._add_regime_features(df)
            print("   ✓ Market regime features added")

        # 6. Microstructure features
        if self.config.USE_MICROSTRUCTURE_FEATURES:
            df = self._add_microstructure_features(df)
            print("   ✓ Microstructure features added")

        # 7. Fractal features
        if self.config.USE_FRACTAL_FEATURES:
            df = self._add_fractal_features(df)
            print("   ✓ Fractal features added")

        # 8. TA features
        if self.config.USE_TA_FEATURES:
            df = self._add_ta_features(df)
            print("   ✓ Technical analysis features added")

        # 9. Statistical features
        df = self._add_statistical_features(df)
        print("   ✓ Statistical features added")

        print()
        return df

    def _add_time_features(self, df):
        """Cyclical time encoding"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Hour
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Day of month
        df['day'] = df['timestamp'].dt.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        # Trading session (example for crypto - 24/7, but can show patterns)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        return df

    def _add_price_action_features(self, df):
        """Price action and momentum features"""

        # Returns
        for period in [1, 3, 5, 10, 20, 30]:
            df[f'return_{period}'] = df['close'].pct_change(period) * 100

        # Log returns (better for ML)
        for period in [1, 5, 10]:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period)) * 100

        # Price position in range
        for period in [10, 20, 50]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            df[f'price_position_{period}'] = (
                (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            )

        # Candle patterns
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-10)

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (
                (df['close'] - df['close'].shift(period)) /
                (df['close'].shift(period) + 1e-10) * 100
            )

        return df

    def _add_volume_features(self, df):
        """Volume-based features"""

        # Volume changes
        df['volume_change'] = df['volume'].pct_change() * 100

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-10)

        # Volume-price features
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10) * 100

        # On-balance volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_sma_20'] = df['obv'].rolling(20).mean()
        df['obv_signal'] = df['obv'] - df['obv_sma_20']

        # Volume profile (simplified)
        for period in [20]:
            df[f'high_volume_price_{period}'] = df.groupby(
                pd.cut(df['close'], bins=self.config.VOLUME_PROFILE_BINS)
            )['volume'].transform('sum')

        # Force Index
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_sma_13'] = df['force_index'].rolling(13).mean()

        # Money Flow
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']

        for period in [10, 20]:
            positive_flow = df['money_flow'].where(df['typical_price'] > df['typical_price'].shift(1), 0)
            negative_flow = df['money_flow'].where(df['typical_price'] < df['typical_price'].shift(1), 0)

            positive_mf = positive_flow.rolling(period).sum()
            negative_mf = negative_flow.rolling(period).sum()

            df[f'mfi_{period}'] = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))

        return df

    def _add_volatility_features(self, df):
        """Volatility and risk features"""

        # Historical volatility
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std() * 100

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / (df['close'] + 1e-10) * 100

        # Bollinger Bands width
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_width_{period}'] = (std * 2) / (sma + 1e-10) * 100
            df[f'bb_position_{period}'] = (df['close'] - sma) / (std + 1e-10)

        # Volatility clustering (GARCH-like)
        returns = df['close'].pct_change()
        for lag in [1, 2, 3, 5]:
            df[f'volatility_lag_{lag}'] = returns.rolling(lag).std() * 100

        # Parkinson's volatility (uses high-low)
        for period in [10, 20]:
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) *
                ((np.log(df['high'] / df['low'])) ** 2).rolling(period).mean()
            ) * 100

        return df

    def _add_regime_features(self, df):
        """Market regime detection features"""

        lookback = self.config.REGIME_LOOKBACK

        # Trend strength (ADX-like)
        for period in [14, 20]:
            plus_dm = df['high'].diff().clip(lower=0)
            minus_dm = -df['low'].diff().clip(upper=0)

            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)

            plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df[f'adx_{period}'] = dx.rolling(period).mean()

            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di

        # Trend direction
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_vs_sma_{period}'] = (
                (df['close'] - df[f'sma_{period}']) / (df[f'sma_{period}'] + 1e-10) * 100
            )

        # EMA
        for period in [9, 21, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Moving average crossovers
        df['ema_9_21_cross'] = df['ema_9'] - df['ema_21']
        df['ema_21_50_cross'] = df['ema_21'] - df['ema_50']

        # Efficiency ratio (Kaufman's)
        for period in [10, 20]:
            change = abs(df['close'] - df['close'].shift(period))
            volatility = abs(df['close'].diff()).rolling(period).sum()
            df[f'efficiency_ratio_{period}'] = change / (volatility + 1e-10)

        # Hurst exponent (trend vs mean reversion)
        for period in [50, 100]:
            if len(df) >= period:
                df[f'hurst_{period}'] = df['close'].rolling(period).apply(
                    lambda x: self._calculate_hurst(x) if len(x) == period else np.nan,
                    raw=True
                )

        return df

    def _calculate_hurst(self, ts):
        """Calculate Hurst exponent (simplified)"""
        try:
            lags = range(2, 20)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5

    def _add_microstructure_features(self, df):
        """Market microstructure features"""

        # Spread
        df['hl_spread'] = (df['high'] - df['low']) / (df['low'] + 1e-10) * 100
        df['hl_spread_ma_10'] = df['hl_spread'].rolling(10).mean()

        # Price impact (simplified)
        df['price_impact'] = abs(df['close'].diff()) / (df['volume'] + 1e-10) * 1e6

        # Order flow imbalance (approximation using volume and price change)
        df['volume_signed'] = df['volume'] * np.sign(df['close'].diff())
        for period in [5, 10, 20]:
            df[f'order_flow_imbalance_{period}'] = df['volume_signed'].rolling(period).sum()

        # Buy/Sell pressure (approximation)
        df['buy_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)) * df['volume']
        df['sell_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)) * df['volume']

        for period in [10, 20]:
            df[f'buy_sell_ratio_{period}'] = (
                df['buy_pressure'].rolling(period).sum() /
                (df['sell_pressure'].rolling(period).sum() + 1e-10)
            )

        # Amihud illiquidity
        for period in [10, 20]:
            df[f'illiquidity_{period}'] = (
                abs(df['close'].pct_change()) / (df['volume'] + 1e-10)
            ).rolling(period).mean() * 1e6

        return df

    def _add_fractal_features(self, df):
        """Fractal and pattern features"""

        for period in self.config.FRACTAL_PERIODS:
            # Fractal highs/lows
            df[f'fractal_high_{period}'] = (
                (df['high'] == df['high'].rolling(period * 2 + 1, center=True).max()).astype(int)
            )
            df[f'fractal_low_{period}'] = (
                (df['low'] == df['low'].rolling(period * 2 + 1, center=True).min()).astype(int)
            )

            # Distance to last fractal
            df[f'bars_since_fractal_high_{period}'] = (
                df.groupby(df[f'fractal_high_{period}'].cumsum()).cumcount()
            )
            df[f'bars_since_fractal_low_{period}'] = (
                df.groupby(df[f'fractal_low_{period}'].cumsum()).cumcount()
            )

        # Pivot points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])

        df['price_vs_pivot'] = (df['close'] - df['pivot']) / (df['pivot'] + 1e-10) * 100

        return df

    def _add_ta_features(self, df):
        """Add standard TA library features (selective)"""

        # Beware: add_all_ta_features creates many features, use selectively
        df_ta = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df_ta = dropna(df_ta)

        # Only add specific indicators to avoid noise
        from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
        from ta.trend import MACD, CCIIndicator, IchimokuIndicator

        # RSI
        for period in [14, 21]:
            rsi = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()

        # Stochastic
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # CCI
        cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20)
        df['cci_20'] = cci.cci()

        # Williams %R
        wr = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14)
        df['williams_r'] = wr.williams_r()

        return df

    def _add_statistical_features(self, df):
        """Statistical features on price and returns"""

        for period in [10, 20, 50]:
            # Rolling statistics on returns
            returns = df['close'].pct_change()

            df[f'return_skew_{period}'] = returns.rolling(period).apply(skew, raw=True)
            df[f'return_kurt_{period}'] = returns.rolling(period).apply(kurtosis, raw=True)
            df[f'return_mean_{period}'] = returns.rolling(period).mean() * 100
            df[f'return_std_{period}'] = returns.rolling(period).std() * 100

            # Rolling Z-score
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)

        return df

    def get_feature_names(self, df, exclude_cols=None):
        """Get list of feature column names"""
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                           'target', 'target_up', 'target_down', 'hour', 'dayofweek', 'day']

        features = [col for col in df.columns if col not in exclude_cols]
        return features
