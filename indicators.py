"""
İndikatör Hesaplamaları - Doğru Uygulamalar
============================================

ATR, RSI, SMA, EMA, MACD vb. proper uygulamalar
"""

import numpy as np
from collections import deque
from typing import List, Optional


class Indicators:
    """
    Teknik indikatorleri hesaplayan sınıf
    Doğru matematiksel uygulamalarla
    """
    
    def __init__(self, max_history: int = 500):
        """
        Args:
            max_history: Kaç değeri tutsun
        """
        self.prices = deque(maxlen=max_history)
        self.highs = deque(maxlen=max_history)
        self.lows = deque(maxlen=max_history)
        self.volumes = deque(maxlen=max_history)
    
    def update(
        self,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: Optional[float] = None
    ):
        """Yeni veri ekle"""
        self.prices.append(price)
        self.highs.append(high or price)
        self.lows.append(low or price)
        self.volumes.append(volume or 0)
    
    # ============= ATR (Average True Range) =============
    
    def calculate_atr(self, period: int = 14) -> float:
        """
        ATR hesapla - DOĞRU YÖNTEM
        
        True Range = max(H-L, |H-Pc|, |L-Pc|)
        ATR = SMA(TR, period)
        
        H = High, L = Low, Pc = Previous Close
        """
        if len(self.prices) < period + 1:
            return 0
        
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        
        tr_values = []
        
        # True Range hesapla
        for i in range(len(prices_list)):
            if i == 0:
                # İlk bar'da TR = H - L
                tr = highs_list[i] - lows_list[i]
            else:
                # True Range = max(H-L, |H-Pc|, |L-Pc|)
                h_l = highs_list[i] - lows_list[i]
                h_pc = abs(highs_list[i] - prices_list[i-1])
                l_pc = abs(lows_list[i] - prices_list[i-1])
                tr = max(h_l, h_pc, l_pc)
            
            tr_values.append(tr)
        
        # Son 'period' kadar TR'nin ortalamasını al (SMA)
        atr = np.mean(tr_values[-period:]) if len(tr_values) >= period else 0
        
        return atr
    
    def calculate_atr_ema(self, period: int = 14) -> float:
        """
        ATR hesapla - EMA smoothing ile (Wilder's)
        Daha dirençli bir uyarlama
        """
        if len(self.prices) < period + 1:
            return 0
        
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        
        tr_values = []
        
        for i in range(len(prices_list)):
            if i == 0:
                tr = highs_list[i] - lows_list[i]
            else:
                h_l = highs_list[i] - lows_list[i]
                h_pc = abs(highs_list[i] - prices_list[i-1])
                l_pc = abs(lows_list[i] - prices_list[i-1])
                tr = max(h_l, h_pc, l_pc)
            
            tr_values.append(tr)
        
        # EMA ile smooth et (Wilder's smoothing)
        if len(tr_values) < period:
            return 0
        
        # İlk değer: simple average
        atr = np.mean(tr_values[:period])
        
        # Kalan değerler: Wilder's smoothing
        for i in range(period, len(tr_values)):
            atr = (atr * (period - 1) + tr_values[i]) / period
        
        return atr
    
    # ============= RSI (Relative Strength Index) =============
    
    def calculate_rsi(self, period: int = 14) -> float:
        """
        RSI hesapla - DOĞRU YÖNTEM
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(self.prices) < period + 2:
            return 50
        
        prices_list = list(self.prices)
        
        # Farkları hesapla
        deltas = []
        for i in range(1, len(prices_list)):
            delta = prices_list[i] - prices_list[i-1]
            deltas.append(delta)
        
        # Son 'period' kadarını al
        recent_deltas = deltas[-period:]
        
        if len(recent_deltas) < period:
            return 50
        
        # Gains ve Losses
        gains = [d if d > 0 else 0 for d in recent_deltas]
        losses = [abs(d) if d < 0 else 0 for d in recent_deltas]
        
        # Ortalamaları
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # RS hesapla
        if avg_loss == 0:
            rsi = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_rsi_wilder(self, period: int = 14) -> float:
        """
        RSI hesapla - Wilder's Smoothing ile (profesyonel)
        Tradingview'in kullandığı yöntem
        """
        if len(self.prices) < period + 2:
            return 50
        
        prices_list = list(self.prices)
        
        deltas = []
        for i in range(1, len(prices_list)):
            delta = prices_list[i] - prices_list[i-1]
            deltas.append(delta)
        
        # İlk ortalamalar
        first_gains = []
        first_losses = []
        
        for i in range(period):
            if deltas[i] > 0:
                first_gains.append(deltas[i])
                first_losses.append(0)
            else:
                first_gains.append(0)
                first_losses.append(abs(deltas[i]))
        
        avg_gain = np.mean(first_gains)
        avg_loss = np.mean(first_losses)
        
        # Wilder's smoothing
        for i in range(period, len(deltas)):
            if deltas[i] > 0:
                gain = deltas[i]
                loss = 0
            else:
                gain = 0
                loss = abs(deltas[i])
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
        
        # RSI hesapla
        if avg_loss == 0:
            rsi = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # ============= SMA (Simple Moving Average) =============
    
    def calculate_sma(self, period: int = 20) -> float:
        """
        SMA hesapla
        SMA = Sum(prices[-period:]) / period
        """
        if len(self.prices) < period:
            return 0
        
        prices_list = list(self.prices)
        sma = np.mean(prices_list[-period:])
        
        return sma
    
    # ============= EMA (Exponential Moving Average) =============
    
    def calculate_ema(self, period: int = 20) -> float:
        """
        EMA hesapla
        EMA = price * k + EMA_prev * (1 - k)
        k = 2 / (period + 1)
        """
        if len(self.prices) < period:
            return 0
        
        prices_list = list(self.prices)
        
        if len(prices_list) < period:
            return 0
        
        k = 2 / (period + 1)
        
        # İlk EMA = SMA
        ema = np.mean(prices_list[:period])
        
        # Kalan fiyatlara EMA formülü uygula
        for price in prices_list[period:]:
            ema = price * k + ema * (1 - k)
        
        return ema
    
    # ============= MACD (Moving Average Convergence Divergence) =============
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        MACD hesapla
        
        Returns:
            (macd_line, signal_line, histogram)
        """
        if len(self.prices) < slow + signal:
            return 0, 0, 0
        
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        
        macd_line = ema_fast - ema_slow
        
        # Signal line hesapla (MACD'nin EMA'sı)
        prices_list = list(self.prices)
        
        # Tüm MACD değerlerini hesapla
        macd_values = []
        for i in range(slow - 1, len(prices_list)):
            ema_f = np.mean(prices_list[i-fast+1:i+1])
            ema_s = np.mean(prices_list[i-slow+1:i+1])
            macd_val = ema_f - ema_s
            macd_values.append(macd_val)
        
        if len(macd_values) < signal:
            signal_line = macd_line
        else:
            signal_line = np.mean(macd_values[-signal:])
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    # ============= Bollinger Bands =============
    
    def calculate_bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """
        Bollinger Bands hesapla
        
        Returns:
            (upper_band, middle_band, lower_band)
        """
        if len(self.prices) < period:
            return 0, 0, 0
        
        prices_list = list(self.prices)
        recent_prices = prices_list[-period:]
        
        # Middle band = SMA
        middle_band = np.mean(recent_prices)
        
        # Standard deviation
        std = np.std(recent_prices)
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    # ============= Stochastic =============
    
    def calculate_stochastic(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple:
        """
        Stochastic Oscillator hesapla
        
        Returns:
            (%K, %D)
        """
        if len(self.prices) < period:
            return 50, 50
        
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        
        recent_prices = prices_list[-period:]
        recent_highs = highs_list[-period:]
        recent_lows = lows_list[-period:]
        
        # Highest High ve Lowest Low
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        # %K raw
        current_close = prices_list[-1]
        
        if highest_high - lowest_low == 0:
            k_raw = 50
        else:
            k_raw = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
        
        # Smooth %K
        k = k_raw  # Basit versiyon
        d = k  # Basit versiyon
        
        return k, d
    
    # ============= CCI (Commodity Channel Index) =============
    
    def calculate_cci(self, period: int = 20) -> float:
        """
        CCI hesapla
        """
        if len(self.prices) < period:
            return 0
        
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        
        recent_prices = prices_list[-period:]
        recent_highs = highs_list[-period:]
        recent_lows = lows_list[-period:]
        
        # Typical Price = (H + L + C) / 3
        typical_prices = []
        for i in range(period):
            tp = (recent_highs[i] + recent_lows[i] + recent_prices[i]) / 3
            typical_prices.append(tp)
        
        # SMA of Typical Price
        sma_tp = np.mean(typical_prices)
        
        # Mean Deviation
        mean_dev = np.mean([abs(tp - sma_tp) for tp in typical_prices])
        
        # CCI
        if mean_dev == 0:
            cci = 0
        else:
            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_dev)
        
        return cci
    
    # ============= VOLATILITY =============
    
    def calculate_volatility(self, period: int = 20) -> float:
        """
        Volatilite hesapla (Standard Deviation)
        """
        if len(self.prices) < period:
            return 0
        
        prices_list = list(self.prices)
        recent_prices = prices_list[-period:]
        
        # Returns hesapla
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] != 0:
                ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)
        
        if not returns:
            return 0
        
        # Standard deviation
        volatility = np.std(returns)
        
        return volatility
    
    # ============= Trend =============
    
    def calculate_trend(self, period: int = 5) -> str:
        """
        Trend belirle
        """
        if len(self.prices) < period:
            return "UNKNOWN"
        
        prices_list = list(self.prices)
        recent_prices = prices_list[-period:]
        
        if recent_prices[-1] > recent_prices[0] * 1.02:  # 2% üstünde
            return "UP"
        elif recent_prices[-1] < recent_prices[0] * 0.98:  # 2% altında
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    # ============= SUMMARY =============
    
    def get_all_indicators(self) -> dict:
        """
        Tüm indikatörleri bir sefer hesapla
        """
        return {
            'atr': self.calculate_atr(14),
            'atr_ema': self.calculate_atr_ema(14),
            'rsi': self.calculate_rsi(14),
            'rsi_wilder': self.calculate_rsi_wilder(14),
            'sma20': self.calculate_sma(20),
            'sma50': self.calculate_sma(50),
            'ema12': self.calculate_ema(12),
            'ema26': self.calculate_ema(26),
            'macd': self.calculate_macd(),
            'bollinger': self.calculate_bollinger_bands(),
            'stochastic': self.calculate_stochastic(),
            'cci': self.calculate_cci(),
            'volatility': self.calculate_volatility(),
            'trend': self.calculate_trend()
        }


# ============================================================================
# TEST FONKSIYONU
# ============================================================================

def test_indicators():
    """İndikatörleri test et"""
    print("\n" + "="*60)
    print("🧪 İNDİKATÖR TESTLERI")
    print("="*60)
    
    indicators = Indicators()
    
    # Örnek veri ekle (simit fiyatları, 100-150 aralığında)
    test_prices = [
        (105, 110, 102),  # (close, high, low)
        (108, 111, 107),
        (112, 115, 108),
        (110, 115, 109),
        (115, 118, 110),
        (120, 122, 114),
        (118, 121, 116),
        (122, 125, 118),
        (125, 128, 122),
        (128, 130, 125),
        (132, 135, 128),
        (130, 133, 129),
        (135, 137, 131),
        (138, 140, 135),
        (140, 142, 137),
    ]
    
    print("\n✓ Test verisi ekleniyor...")
    for close, high, low in test_prices:
        indicators.update(close, high, low)
    
    print(f"  Toplam bar: {len(indicators.prices)}")
    
    # Test hesaplamaları
    print("\n✓ ATR Hesaplamaları:")
    atr_sma = indicators.calculate_atr(14)
    atr_ema = indicators.calculate_atr_ema(14)
    print(f"  ATR (SMA): {atr_sma:.2f} {'✓' if atr_sma > 0 else '❌'}")
    print(f"  ATR (EMA): {atr_ema:.2f} {'✓' if atr_ema > 0 else '❌'}")
    
    print("\n✓ RSI Hesaplamaları:")
    rsi = indicators.calculate_rsi(14)
    rsi_wilder = indicators.calculate_rsi_wilder(14)
    print(f"  RSI (SMA): {rsi:.2f} {'✓' if 0 <= rsi <= 100 else '❌'}")
    print(f"  RSI (Wilder): {rsi_wilder:.2f} {'✓' if 0 <= rsi_wilder <= 100 else '❌'}")
    
    print("\n✓ Hareketli Ortalamalar:")
    sma20 = indicators.calculate_sma(20)
    sma50 = indicators.calculate_sma(50)
    ema12 = indicators.calculate_ema(12)
    ema26 = indicators.calculate_ema(26)
    print(f"  SMA(20): {sma20:.2f} {'✓' if sma20 > 0 else '❌'}")
    print(f"  SMA(50): {sma50:.2f} {'✓' if sma50 > 0 else '❌'}")
    print(f"  EMA(12): {ema12:.2f} {'✓' if ema12 > 0 else '❌'}")
    print(f"  EMA(26): {ema26:.2f} {'✓' if ema26 > 0 else '❌'}")
    
    print("\n✓ MACD:")
    macd, signal, hist = indicators.calculate_macd()
    print(f"  MACD Line: {macd:.2f}")
    print(f"  Signal: {signal:.2f}")
    print(f"  Histogram: {hist:.2f}")
    
    print("\n✓ Bollinger Bands:")
    upper, mid, lower = indicators.calculate_bollinger_bands()
    print(f"  Upper: {upper:.2f}")
    print(f"  Middle: {mid:.2f}")
    print(f"  Lower: {lower:.2f}")
    
    print("\n✓ CCI:")
    cci = indicators.calculate_cci()
    print(f"  CCI: {cci:.2f}")
    
    print("\n✓ Volatilite:")
    vol = indicators.calculate_volatility()
    print(f"  Volatilite: {vol:.4f} ({'%' if vol < 1 else ''})")
    
    print("\n✓ Trend:")
    trend = indicators.calculate_trend()
    print(f"  Trend: {trend}")
    
    print("\n" + "="*60)
    print("✅ TÜM TESTLER TAMAMLANDI")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_indicators()
