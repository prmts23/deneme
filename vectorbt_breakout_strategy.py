"""
VectorBT Breakout Strategy - FreqTrade Implementation
=====================================================

Strateji Mantığı:
1. Önceki günün yüksek (Y_HH) ve düşük (Y_LL) seviyelerini hesapla
2. 4H timeframe'de trend filtresi uygula (rolling max/min ile)
3. Long: Trend yukarı + close > Y_HH breakout
4. Short: Trend aşağı + close < Y_LL breakdown
5. SL: %1.5, TP: %7.0

VectorBT script ile %100 aynı sonuçları verecek şekilde tasarlanmıştır.
"""

from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class VectorBTBreakoutStrategy(IStrategy):
    """
    VectorBT backtest scripti ile aynı mantığı kullanan FreqTrade stratejisi.

    Parametreler:
    - stoploss: %1.5 (-0.015)
    - take_profit: %7.0 (0.070)
    - lookback_4h: 6 bar (4 saatlik)
    """

    # Strateji Parametreleri
    INTERFACE_VERSION = 3

    # Optimal parametreler (VectorBT backtest sonuçlarından)
    lookback_4h = 6  # 4 saatlik bar sayısı (rollling pencere)

    # Risk Yönetimi
    stoploss = -0.015  # %1.5 stop loss

    # Take Profit - ROI tablosu
    minimal_roi = {
        "0": 0.070,  # %7.0 take profit
    }

    # Trailing stop kullanma (VectorBT scriptinde yok)
    trailing_stop = False

    # Timeframe
    timeframe = '5m'

    # Startup candle count (en fazla ihtiyaç duyulan candle sayısı)
    # 1 gün = 288 candle (5m), güvenli olması için 500
    startup_candle_count: int = 500

    # Strateji çalışma modu
    can_short = True  # Short işlemlere izin ver

    # Order types
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Position adjustment ayarları (DCA vs. için, burada kullanmıyoruz)
    position_adjustment_enable = False

    # ============================================
    # INFORMATIVE TIMEFRAMES
    # ============================================

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        4 saatlik timeframe için trend filtresi hesaplaması.

        Mantık:
        1. Rolling lookback window ile max/min hesapla
        2. Close > roll_max => trend = 1 (long)
        3. Close < roll_min => trend = -1 (short)
        4. Forward fill ile trendi taşı
        """

        # Rolling max/min hesaplama (shift ile future leak önleme)
        dataframe['roll_max'] = dataframe['high'].rolling(window=self.lookback_4h).max().shift(1)
        dataframe['roll_min'] = dataframe['low'].rolling(window=self.lookback_4h).min().shift(1)

        # Trend belirleme
        dataframe['trend'] = 0
        dataframe.loc[dataframe['close'] > dataframe['roll_max'], 'trend'] = 1
        dataframe.loc[dataframe['close'] < dataframe['roll_min'], 'trend'] = -1

        # Sıfırları forward fill ile doldur (VectorBT scriptindeki mantık)
        dataframe['trend'] = dataframe['trend'].replace(0, np.nan).ffill().fillna(0)

        return dataframe

    # ============================================
    # MAIN INDICATORS
    # ============================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        5 dakikalık timeframe için indikatör hesaplamaları.

        Hesaplamalar:
        1. Önceki günün high/low seviyelerini bul (Y_HH, Y_LL)
        2. 4H trend filtresini merge et
        """

        # 1. Önceki Günün High/Low Hesaplama
        # ---------------------------------------------------------
        # VectorBT scriptindeki mantık:
        # - 5m'yi 1h'ye resample et
        # - 1h'yi günlüğe resample et (high max, low min)
        # - shift(1) ile önceki güne git
        # - 5m'ye forward fill ile geri getir

        # Günlük high/low hesaplamak için tarih bilgisi gerekli
        df_temp = dataframe.copy()
        df_temp['date_only'] = pd.to_datetime(df_temp['date']).dt.normalize()

        # Günlük max/min hesapla
        daily_stats = df_temp.groupby('date_only').agg({
            'high': 'max',
            'low': 'min'
        }).rename(columns={'high': 'daily_high', 'low': 'daily_low'})

        # Önceki güne shift et
        daily_stats['Y_HH'] = daily_stats['daily_high'].shift(1)
        daily_stats['Y_LL'] = daily_stats['daily_low'].shift(1)

        # Ana dataframe'e merge et
        df_temp = df_temp.merge(
            daily_stats[['Y_HH', 'Y_LL']],
            left_on='date_only',
            right_index=True,
            how='left'
        )

        # Forward fill (günün başında değer olana kadar taşı)
        dataframe['Y_HH'] = df_temp['Y_HH'].ffill()
        dataframe['Y_LL'] = df_temp['Y_LL'].ffill()

        # 2. 4H Trend Filtresini Merge Et
        # ---------------------------------------------------------
        # @informative decorator otomatik olarak merge eder
        # Sütun adı: trend_4h

        return dataframe

    # ============================================
    # ENTRY SIGNALS
    # ============================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Giriş sinyallerini belirle.

        LONG Entry:
        - 4H trend = 1 (yukarı)
        - Close > Y_HH (önceki günün yüksek seviyesi)
        - Previous close <= Y_HH (breakout anı)

        SHORT Entry:
        - 4H trend = -1 (aşağı)
        - Close < Y_LL (önceki günün düşük seviyesi)
        - Previous close >= Y_LL (breakdown anı)
        - Long sinyali yoksa (çakışma önleme)
        """

        # Previous close
        dataframe['prev_close'] = dataframe['close'].shift(1)

        # LONG CONDITIONS
        long_condition = (
            (dataframe['trend_4h'] == 1) &  # 4H trend yukarı
            (dataframe['close'] > dataframe['Y_HH']) &  # Breakout
            (dataframe['prev_close'] <= dataframe['Y_HH'])  # İlk breakout anı
        )

        # SHORT CONDITIONS
        short_condition = (
            (dataframe['trend_4h'] == -1) &  # 4H trend aşağı
            (dataframe['close'] < dataframe['Y_LL']) &  # Breakdown
            (dataframe['prev_close'] >= dataframe['Y_LL'])  # İlk breakdown anı
        )

        # Çakışma önleme (VectorBT scriptindeki mantık)
        short_condition = short_condition & (~long_condition)

        # Sinyal ata
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1

        return dataframe

    # ============================================
    # EXIT SIGNALS
    # ============================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Çıkış sinyalleri.

        VectorBT scriptinde sadece SL/TP ile çıkış yapılıyor,
        trend değişimi ile çıkış yok. Bu yüzden exit sinyali vermeye gerek yok.
        SL ve ROI (take profit) otomatik olarak işlem görecek.
        """

        # Exit sinyali yok, sadece SL/TP kullan
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe

    # ============================================
    # CUSTOM STOPLOSS (Opsiyonel)
    # ============================================

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss implementasyonu.

        VectorBT scriptinde sabit %1.5 SL kullanılıyor,
        bu yüzden custom logic'e gerek yok. Strateji seviyesindeki
        stoploss = -0.015 yeterli.

        Return None: Strateji seviyesindeki stoploss'u kullan
        """
        return None

    # ============================================
    # CUSTOM EXIT (Opsiyonel)
    # ============================================

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                   current_rate: float, current_profit: float, **kwargs) -> str:
        """
        Custom exit logic.

        VectorBT scriptinde sadece SL/TP var, özel çıkış mantığı yok.
        """
        return None


# ============================================
# HYPEROPT PARAMETERS (Opsiyonel Optimizasyon)
# ============================================

"""
Eğer FreqTrade Hyperopt ile optimize etmek isterseniz:

from freqtrade.optimize.space import Categorical, DecimalParameter, IntParameter

class VectorBTBreakoutStrategy(IStrategy):

    # Stop Loss Optimization
    stoploss = DecimalParameter(-0.030, -0.005, default=-0.015, decimals=3, space='sell')

    # Take Profit Optimization
    roi_tp = DecimalParameter(0.030, 0.070, default=0.070, decimals=3, space='sell')

    # Lookback Optimization
    lookback_4h = IntParameter(3, 11, default=6, space='buy')

    minimal_roi = {
        "0": roi_tp.value,
    }
"""
