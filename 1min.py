import asyncio
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional
import numpy as np
from btcturk_clean import BTCTurkTradeFeed
from telegram_notifier import TelegramNotifier, AlertLevel
from trade_stats import TradeStats
from indicators import Indicators


class MinuteBarBuilder:
    """1-minute OHLCV + CVD bar'ları"""
    
    def __init__(self):
        self.current_minute = None
        self.bars = deque(maxlen=500)
        
        self.bar_open = None
        self.bar_high = None
        self.bar_low = None
        self.bar_close = None
        self.bar_volume = 0
        self.bar_trades_count = 0
        self.buy_volume = 0
        self.sell_volume = 0
    
    def process_trade(self, price: float, side: str, amount: float, timestamp: str) -> Optional[Dict]:
        """Trade'i işle"""
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        minute = dt.replace(second=0, microsecond=0)
        
        # Yeni dakika - bar kapat
        if self.current_minute is not None and minute > self.current_minute:
            completed_bar = self._close_bar()
            self._start_new_bar(minute, price)
            return completed_bar
        
        # İlk veri
        if self.current_minute is None:
            self._start_new_bar(minute, price)
        
        # Bar'ı güncelle
        self.bar_high = max(self.bar_high, price)
        self.bar_low = min(self.bar_low, price)
        self.bar_close = price
        self.bar_volume += amount
        self.bar_trades_count += 1
        
        if side == "BUY":
            self.buy_volume += amount
        else:
            self.sell_volume += amount
        
        return None
    
    def _start_new_bar(self, minute: datetime, price: float):
        self.current_minute = minute
        self.bar_open = price
        self.bar_high = price
        self.bar_low = price
        self.bar_close = price
        self.bar_volume = 0
        self.bar_trades_count = 0
        self.buy_volume = 0
        self.sell_volume = 0
    
    def _close_bar(self) -> Dict:
        """Bar'ı kapat"""
        # CVD Delta = Buy Volume - Sell Volume
        cvd_delta = self.buy_volume - self.sell_volume
        
        bar = {
            'time': self.current_minute,
            'open': self.bar_open,
            'high': self.bar_high,
            'low': self.bar_low,
            'close': self.bar_close,
            'volume': self.bar_volume,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'cvd_delta': cvd_delta,  # ← YENİ!
            'trades': self.bar_trades_count,
        }
        self.bars.append(bar)
        return bar
    
    def get_closed_bars(self) -> List[Dict]:
        return list(self.bars)


class CVDIndicator:
    """CVD (Cumulative Volume Delta) hesaplayıcı"""
    
    def __init__(self, max_history: int = 300):
        self.cvd_values = deque(maxlen=max_history)
        self.cumulative_cvd = 0
    
    def add_bar(self, buy_volume: float, sell_volume: float):
        """Bar ekle ve CVD hesapla"""
        delta = buy_volume - sell_volume
        self.cumulative_cvd += delta
        self.cvd_values.append({
            'delta': delta,
            'cumulative': self.cumulative_cvd
        })
    
    def get_cvd_delta(self) -> float:
        """Son bar'ın CVD delta'sı"""
        if not self.cvd_values:
            return 0
        return self.cvd_values[-1]['delta']
    
    def get_cumulative_cvd(self) -> float:
        """Kümülatif CVD"""
        return self.cumulative_cvd
    
    def get_cvd_momentum(self, period: int = 5) -> float:
        """CVD momentum - son X bar'ın CVD değişimi"""
        if len(self.cvd_values) < period:
            return 0
        
        recent_cvds = list(self.cvd_values)[-period:]
        total_delta = sum(v['delta'] for v in recent_cvds)
        return total_delta
    
    def get_cvd_trend(self, period: int = 10) -> str:
        """CVD trend'i belirle"""
        if len(self.cvd_values) < period:
            return "NEUTRAL"
        
        recent_cvds = list(self.cvd_values)[-period:]
        momentum = sum(v['delta'] for v in recent_cvds)
        
        if momentum > 0:
            return "BULLISH"
        elif momentum < 0:
            return "BEARISH"
        else:
            return "NEUTRAL"


class PairStrategyWithCVD:
    """1-minute bars + CVD ile strateji - DÜZELTILMIŞ"""

    def __init__(
        self,
        pair: str,
        notifier: Optional[TelegramNotifier] = None,
        stats_report_interval: int = 3600
    ):
        self.pair = pair
        self.notifier = notifier
        self.stats_report_interval = stats_report_interval

        # Bar builder
        self.bar_builder = MinuteBarBuilder()

        # Indicators
        self.indicators = Indicators(max_history=300)
        self.cvd = CVDIndicator(max_history=300)

        # Strategy state
        self.position = None  # "LONG" or "SHORT" or None
        self.entry_price = None
        self.entry_time = None
        self.entry_bar_index = None
        self.entry_cvd_trend = None

        # ✅ YENİ: Entry anındaki değerleri sakla (değişmesin!)
        self.entry_atr = None
        self.stop_loss = None
        self.take_profit = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None

        # Stats
        self.stats = TradeStats(max_history=1000)
        self.last_stats_report = datetime.now()

        # Technical Parameters
        self.atr_period = 14
        self.rsi_period = 14
        self.win_rate_threshold = 0.49

        # CVD Parameters
        self.cvd_momentum_period = 5
        self.cvd_delta_threshold = 0.5

        # ✅ YENİ: SL/TP Multipliers - Basit ve net
        self.sl_atr_multiplier = 2.0   # Stop Loss: 2×ATR
        self.tp_atr_multiplier = 3.0   # Take Profit: 3×ATR (Risk/Reward = 1:1.5)
        self.min_risk_reward = 1.5     # Minimum R/R oranı

        # ✅ YENİ: Trailing Stop
        self.use_trailing_stop = True
        self.trailing_stop_trigger = 2.0  # 2×ATR kar sonra trailing aktif
        self.trailing_stop_distance = 1.0 # 1×ATR mesafede trailing SL

        # ✅ YENİ: Fees & Slippage (BTCTurk)
        self.maker_fee = 0.0008   # 0.08%
        self.taker_fee = 0.0016   # 0.16%
        self.slippage = 0.0005    # 0.05% tahmini slippage

        # Entry Signal Parameters
        self.rsi_oversold = 35    # RSI < 35 → oversold (LONG)
        self.rsi_overbought = 65  # RSI > 65 → overbought (SHORT)
        self.cvd_momentum_threshold = 2.0  # Minimum CVD momentum
        self.buy_ratio_threshold = 0.60    # Buy ratio > 60% (LONG)
        self.sell_ratio_threshold = 0.40   # Buy ratio < 40% (SHORT)

        print(f"✅ {pair} strategy başlatıldı (DÜZELTILMIŞ 1-MIN + CVD)")
        print(f"   SL: {self.sl_atr_multiplier}×ATR | TP: {self.tp_atr_multiplier}×ATR | R/R: 1:{self.tp_atr_multiplier/self.sl_atr_multiplier:.1f}")
        print(f"   Fees: {self.taker_fee:.2%} | Trailing Stop: {'✅' if self.use_trailing_stop else '❌'}")
    
    async def on_trade(self, price: float, side: str, amount: float, timestamp: str):
        """Trade'i işle"""
        closed_bar = self.bar_builder.process_trade(price, side, amount, timestamp)
        
        if closed_bar:
            await self._process_closed_bar(closed_bar)
        
        await self._check_open_position(price)
    
    async def _process_closed_bar(self, bar: Dict):
        """Kapalı bar'ı işle"""
        # Indicators'a ekle
        self.indicators.update(
            price=bar['close'],
            high=bar['high'],
            low=bar['low'],
            volume=bar['volume']
        )
        
        # CVD'ye ekle
        self.cvd.add_bar(bar['buy_volume'], bar['sell_volume'])
        
        # Yeterli veri?
        if len(self.indicators.prices) < self.atr_period + 2:
            #print(f"\n⏳ [{self.pair}] Bar #{len(self.indicators.prices)} - Veri toplanıyor...")
            return
        
        # İndikatörleri hesapla
        atr = self.indicators.calculate_atr(self.atr_period)
        rsi = self.indicators.calculate_rsi_wilder(self.rsi_period)
        sma20 = self.indicators.calculate_sma(20)
        sma50 = self.indicators.calculate_sma(50)
        win_rate = self._calculate_win_rate()
        
        # CVD metriklerini hesapla
        cvd_delta = self.cvd.get_cvd_delta()
        cvd_momentum = self.cvd.get_cvd_momentum(self.cvd_momentum_period)
        cvd_trend = self.cvd.get_cvd_trend(10)
        cumulative_cvd = self.cvd.get_cumulative_cvd()
        
        # CVD ratio (buy_volume / total_volume)
        total_vol = bar['buy_volume'] + bar['sell_volume']
        buy_ratio = bar['buy_volume'] / total_vol if total_vol > 0 else 0.5
        
        # Debug output
        print(f"\n✓ [{self.pair}] Bar #{len(self.indicators.prices)} KAPALANDI")
        print(f"  Time: {bar['time']}")
        print(f"  OHLC: {bar['open']:.2f} / {bar['high']:.2f} / {bar['low']:.2f} / {bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:.2f} | Buy: {bar['buy_volume']:.2f} | Sell: {bar['sell_volume']:.2f}")
        print(f"  CVD Delta: {cvd_delta:+.2f} | Momentum: {cvd_momentum:+.2f} | Trend: {cvd_trend}")
        print(f"  Buy Ratio: {buy_ratio:.1%}")
        print(f"  ATR({self.atr_period}): {atr:.4f} | RSI({self.rsi_period}): {rsi:.1f}")
        print(f"  SMA20: {sma20:.2f} | SMA50: {sma50:.2f} | WR: {win_rate:.1%}")
        
        # Sinyal
        await self._check_signals(
            bar['close'], atr, rsi, sma20, sma50, win_rate,
            cvd_delta, cvd_momentum, cvd_trend, buy_ratio
        )
        
        # Stats report
        await self._check_stats_report()
    
    async def _check_open_position(self, current_price: float):
        """✅ DÜZELTILMIŞ: Açık pozisyon kontrol - Sabit SL/TP + Trailing Stop"""
        if self.position is None:
            return

        # Güvenlik: Entry değerleri yoksa çık
        if self.stop_loss is None or self.take_profit is None or self.entry_atr is None:
            print(f"⚠️ [{self.pair}] WARNING: Position açık ama SL/TP tanımlı değil! Pozisyon kapatılıyor.")
            await self._close_position(current_price, "EMERGENCY EXIT")
            return

        # Trailing Stop Logic
        if self.use_trailing_stop and self.entry_atr > 0:
            if self.position == "LONG":
                # Track highest price
                if self.highest_price_since_entry is None:
                    self.highest_price_since_entry = current_price
                else:
                    self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)

                # Kar yeterli mi? (2×ATR)
                profit = current_price - self.entry_price
                trigger_distance = self.trailing_stop_trigger * self.entry_atr

                if profit > trigger_distance:
                    # Trailing SL hesapla
                    trailing_sl = self.highest_price_since_entry - (self.trailing_stop_distance * self.entry_atr)

                    # Trailing SL, orijinal SL'den yüksekse güncelle
                    if trailing_sl > self.stop_loss:
                        old_sl = self.stop_loss
                        self.stop_loss = trailing_sl
                        print(f"   📈 Trailing SL: {old_sl:.2f} → {self.stop_loss:.2f} (High: {self.highest_price_since_entry:.2f})")

            elif self.position == "SHORT":
                # Track lowest price
                if self.lowest_price_since_entry is None:
                    self.lowest_price_since_entry = current_price
                else:
                    self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)

                # Kar yeterli mi? (2×ATR)
                profit = self.entry_price - current_price
                trigger_distance = self.trailing_stop_trigger * self.entry_atr

                if profit > trigger_distance:
                    # Trailing SL hesapla (SHORT için yukarı)
                    trailing_sl = self.lowest_price_since_entry + (self.trailing_stop_distance * self.entry_atr)

                    # Trailing SL, orijinal SL'den düşükse güncelle
                    if trailing_sl < self.stop_loss:
                        old_sl = self.stop_loss
                        self.stop_loss = trailing_sl
                        print(f"   📉 Trailing SL: {old_sl:.2f} → {self.stop_loss:.2f} (Low: {self.lowest_price_since_entry:.2f})")

        # SL/TP Check
        if self.position == "LONG":
            if current_price >= self.take_profit:
                await self._close_position(current_price, "TAKE PROFIT")
            elif current_price <= self.stop_loss:
                await self._close_position(current_price, "STOP LOSS")

        elif self.position == "SHORT":
            if current_price <= self.take_profit:
                await self._close_position(current_price, "TAKE PROFIT")
            elif current_price >= self.stop_loss:
                await self._close_position(current_price, "STOP LOSS")
    
    async def _check_signals(
        self,
        current_price: float,
        atr: float,
        rsi: float,
        sma20: float,
        sma50: float,
        win_rate: float,
        cvd_delta: float,
        cvd_momentum: float,
        cvd_trend: str,
        buy_ratio: float
    ):
        """✅ DÜZELTILMIŞ: Sinyal kontrolü - LONG/SHORT + Fee Check"""

        if win_rate < self.win_rate_threshold:
            return

        if atr <= 0:
            return

        # Pozisyon yoksa sinyal ara
        if self.position is None:
            # ============= LONG SIGNAL =============
            rsi_ok_long = rsi < self.rsi_oversold
            price_ok_long = current_price > sma20 and sma20 > sma50  # Uptrend
            cvd_ok_long = (
                cvd_momentum > self.cvd_momentum_threshold and
                buy_ratio > self.buy_ratio_threshold and
                cvd_trend == "BULLISH"
            )

            # ============= SHORT SIGNAL =============
            rsi_ok_short = rsi > self.rsi_overbought
            price_ok_short = current_price < sma20 and sma20 < sma50  # Downtrend
            cvd_ok_short = (
                cvd_momentum < -self.cvd_momentum_threshold and
                buy_ratio < self.sell_ratio_threshold and
                cvd_trend == "BEARISH"
            )

            # LONG Entry
            if rsi_ok_long and price_ok_long and cvd_ok_long:
                await self._open_long(current_price, atr, rsi, cvd_trend, cvd_momentum, buy_ratio, sma20, sma50, cvd_delta)

            # SHORT Entry
            elif rsi_ok_short and price_ok_short and cvd_ok_short:
                await self._open_short(current_price, atr, rsi, cvd_trend, cvd_momentum, buy_ratio, sma20, sma50, cvd_delta)

    async def _open_long(self, current_price, atr, rsi, cvd_trend, cvd_momentum, buy_ratio, sma20, sma50, cvd_delta):
        """✅ LONG pozisyon aç - Fee kontrolü ile"""
        # Entry anındaki ATR'yi sakla
        self.entry_atr = atr

        # SL/TP hesapla (SABİT - entry anında)
        self.stop_loss = current_price - (self.sl_atr_multiplier * atr)
        self.take_profit = current_price + (self.tp_atr_multiplier * atr)

        # Risk/Reward
        risk = current_price - self.stop_loss
        reward = self.take_profit - current_price
        risk_reward = reward / risk if risk > 0 else 0

        # Minimum R/R kontrolü
        if risk_reward < self.min_risk_reward:
            print(f"   ⚠️ R/R ({risk_reward:.2f}) < Min R/R ({self.min_risk_reward:.2f}) - LONG İPTAL")
            return

        # Fee + Slippage kontrolü
        total_fee = (self.taker_fee * 2) + (self.slippage * 2)  # Entry + Exit
        min_profit_pct = total_fee + 0.001  # +%0.1 güvenlik marjı
        expected_profit_pct = reward / current_price

        if expected_profit_pct < min_profit_pct:
            print(f"   ⚠️ Expected profit ({expected_profit_pct:.2%}) < Min profit ({min_profit_pct:.2%}) - LONG İPTAL")
            return

        # Pozisyon aç
        self.position = "LONG"
        self.entry_price = current_price
        self.entry_time = datetime.now()
        self.entry_bar_index = len(self.indicators.prices)
        self.entry_cvd_trend = cvd_trend
        self.highest_price_since_entry = current_price

        print(f"\n🟢 [{self.pair}] LONG ENTRY @ {current_price:.2f}")
        print(f"   ATR: {atr:.4f} (FIXED) | RSI: {rsi:.1f}")
        print(f"   SL: {self.stop_loss:.2f} (-{self.sl_atr_multiplier}×ATR) | TP: {self.take_profit:.2f} (+{self.tp_atr_multiplier}×ATR)")
        print(f"   R:R: 1:{risk_reward:.2f} | Expected Profit: {expected_profit_pct:.2%} (Min: {min_profit_pct:.2%})")
        print(f"   CVD: {cvd_trend} | Momentum: {cvd_momentum:+.2f} | Buy: {buy_ratio:.1%}")

        if self.notifier:
            message = f"""
<b>🟢 LONG ENTRY - {self.pair}</b>

<b>Price:</b> {current_price:.2f}
<b>Stop Loss:</b> {self.stop_loss:.2f} (-{self.sl_atr_multiplier}×ATR)
<b>Take Profit:</b> {self.take_profit:.2f} (+{self.tp_atr_multiplier}×ATR)
<b>Risk/Reward:</b> 1:{risk_reward:.2f}

<b>Indicators:</b>
├─ RSI: {rsi:.1f}
├─ ATR: {atr:.4f} (FIXED)
├─ SMA20: {sma20:.2f}
└─ SMA50: {sma50:.2f}

<b>Volume Analysis:</b>
├─ CVD Trend: {cvd_trend}
├─ CVD Momentum: {cvd_momentum:+.2f}
├─ Buy Ratio: {buy_ratio:.1%}
└─ CVD Delta: {cvd_delta:+.2f}

<b>Fees:</b> {total_fee:.2%} | <b>Expected:</b> {expected_profit_pct:.2%}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)

    async def _open_short(self, current_price, atr, rsi, cvd_trend, cvd_momentum, buy_ratio, sma20, sma50, cvd_delta):
        """✅ SHORT pozisyon aç - Fee kontrolü ile"""
        # Entry anındaki ATR'yi sakla
        self.entry_atr = atr

        # SL/TP hesapla (SHORT için ters yönde)
        self.stop_loss = current_price + (self.sl_atr_multiplier * atr)
        self.take_profit = current_price - (self.tp_atr_multiplier * atr)

        # Risk/Reward
        risk = self.stop_loss - current_price
        reward = current_price - self.take_profit
        risk_reward = reward / risk if risk > 0 else 0

        # Minimum R/R kontrolü
        if risk_reward < self.min_risk_reward:
            print(f"   ⚠️ R/R ({risk_reward:.2f}) < Min R/R ({self.min_risk_reward:.2f}) - SHORT İPTAL")
            return

        # Fee + Slippage kontrolü
        total_fee = (self.taker_fee * 2) + (self.slippage * 2)
        min_profit_pct = total_fee + 0.001
        expected_profit_pct = reward / current_price

        if expected_profit_pct < min_profit_pct:
            print(f"   ⚠️ Expected profit ({expected_profit_pct:.2%}) < Min profit ({min_profit_pct:.2%}) - SHORT İPTAL")
            return

        # Pozisyon aç
        self.position = "SHORT"
        self.entry_price = current_price
        self.entry_time = datetime.now()
        self.entry_bar_index = len(self.indicators.prices)
        self.entry_cvd_trend = cvd_trend
        self.lowest_price_since_entry = current_price

        print(f"\n🔴 [{self.pair}] SHORT ENTRY @ {current_price:.2f}")
        print(f"   ATR: {atr:.4f} (FIXED) | RSI: {rsi:.1f}")
        print(f"   SL: {self.stop_loss:.2f} (+{self.sl_atr_multiplier}×ATR) | TP: {self.take_profit:.2f} (-{self.tp_atr_multiplier}×ATR)")
        print(f"   R:R: 1:{risk_reward:.2f} | Expected Profit: {expected_profit_pct:.2%} (Min: {min_profit_pct:.2%})")
        print(f"   CVD: {cvd_trend} | Momentum: {cvd_momentum:+.2f} | Buy: {buy_ratio:.1%}")

        if self.notifier:
            message = f"""
<b>🔴 SHORT ENTRY - {self.pair}</b>

<b>Price:</b> {current_price:.2f}
<b>Stop Loss:</b> {self.stop_loss:.2f} (+{self.sl_atr_multiplier}×ATR)
<b>Take Profit:</b> {self.take_profit:.2f} (-{self.tp_atr_multiplier}×ATR)
<b>Risk/Reward:</b> 1:{risk_reward:.2f}

<b>Indicators:</b>
├─ RSI: {rsi:.1f}
├─ ATR: {atr:.4f} (FIXED)
├─ SMA20: {sma20:.2f}
└─ SMA50: {sma50:.2f}

<b>Volume Analysis:</b>
├─ CVD Trend: {cvd_trend}
├─ CVD Momentum: {cvd_momentum:+.2f}
├─ Buy Ratio: {buy_ratio:.1%}
└─ CVD Delta: {cvd_delta:+.2f}

<b>Fees:</b> {total_fee:.2%} | <b>Expected:</b> {expected_profit_pct:.2%}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)
    
    async def _close_position(self, exit_price: float, exit_type: str):
        """✅ DÜZELTILMIŞ: Pozisyonu kapat - NET PnL hesaplama (Fee + Slippage)"""

        # Güvenlik kontrolü
        if self.entry_price is None or self.entry_time is None:
            print(f"⚠️ [{self.pair}] ERROR: Entry bilgileri eksik!")
            self.position = None
            return

        # GROSS PnL (fee olmadan)
        if self.position == "LONG":
            gross_pnl = exit_price - self.entry_price
        elif self.position == "SHORT":
            gross_pnl = self.entry_price - exit_price
        else:
            gross_pnl = 0

        gross_pnl_pct = (gross_pnl / self.entry_price) * 100

        # ✅ NET PnL (Fee + Slippage dahil)
        # Entry: price × (1 + taker_fee + slippage)
        # Exit: price × (1 - taker_fee - slippage)
        if self.position == "LONG":
            entry_cost = self.entry_price * (1 + self.taker_fee + self.slippage)
            exit_revenue = exit_price * (1 - self.taker_fee - self.slippage)
            net_pnl = exit_revenue - entry_cost
        elif self.position == "SHORT":
            entry_revenue = self.entry_price * (1 - self.taker_fee - self.slippage)
            exit_cost = exit_price * (1 + self.taker_fee + self.slippage)
            net_pnl = entry_revenue - exit_cost
        else:
            net_pnl = 0

        net_pnl_pct = (net_pnl / self.entry_price) * 100
        fee_cost = gross_pnl - net_pnl
        fee_cost_pct = (fee_cost / self.entry_price) * 100

        duration = (datetime.now() - self.entry_time).total_seconds()
        bars_held = len(self.indicators.prices) - self.entry_bar_index if self.entry_bar_index else 0

        emoji = "🟢" if net_pnl > 0 else "🔴"
        pos_type = self.position if self.position else "UNKNOWN"

        print(f"\n{emoji} [{self.pair}] {pos_type} EXIT ({exit_type}) @ {exit_price:.2f}")
        print(f"   Entry: {self.entry_price:.2f} | Exit: {exit_price:.2f}")
        print(f"   GROSS PnL: {gross_pnl:+.2f} ({gross_pnl_pct:+.2f}%)")
        print(f"   NET PnL: {net_pnl:+.2f} ({net_pnl_pct:+.2f}%) ← FEES INCLUDED")
        print(f"   Fee Cost: {fee_cost:.2f} ({fee_cost_pct:.2%})")
        print(f"   SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")
        print(f"   Bars: {bars_held} | Duration: {duration:.0f}s")

        # Stats'e NET PnL ekle
        self.stats.add_trade(
            entry_price=self.entry_price,
            exit_price=exit_price,
            pnl=net_pnl,  # ← NET PnL!
            pnl_pct=net_pnl_pct,  # ← NET PnL %!
            duration_sec=duration,
            trade_type=pos_type,
            entry_time=self.entry_time,
            exit_time=datetime.now()
        )

        if self.notifier:
            message = f"""
<b>{emoji} {pos_type} EXIT - {self.pair}</b>

<b>Entry:</b> {self.entry_price:.2f}
<b>Exit:</b> {exit_price:.2f}
<b>Exit Type:</b> {exit_type}

<b>PnL (GROSS):</b> {gross_pnl:+.2f} ({gross_pnl_pct:+.2f}%)
<b>PnL (NET):</b> {net_pnl:+.2f} ({net_pnl_pct:+.2f}%)
<b>Fee Cost:</b> {fee_cost:.2f} ({fee_cost_pct:.2%})

<b>Duration:</b> {duration:.0f}s ({bars_held} bars)
<b>SL:</b> {self.stop_loss:.2f} | <b>TP:</b> {self.take_profit:.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)

        # ✅ Pozisyonu tamamen sıfırla
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_bar_index = None
        self.entry_cvd_trend = None
        self.entry_atr = None
        self.stop_loss = None
        self.take_profit = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
    
    async def _check_stats_report(self):
        now = datetime.now()
        if (now - self.last_stats_report).total_seconds() >= self.stats_report_interval:
            await self._send_stats_report()
            self.last_stats_report = now
    
    async def _send_stats_report(self):
        if not self.notifier or self.stats.total_trades() == 0:
            return
        
        summary = self.stats.get_summary()
        
        message = f"""
<b>📊 {self.pair} - İSTATİSTİKLER (CVD Strategy)</b>

<b>Trades:</b> {summary['total_trades']}
<b>Win Rate:</b> {summary['win_rate']:.1%}
<b>Avg Win:</b> {summary['avg_win']:+.2f}
<b>Total PnL:</b> {summary['total_pnl']:+.2f}
<b>Sharpe:</b> {summary['sharpe_ratio']:.2f}
<b>Max DD:</b> {summary['max_drawdown']:+.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
        await self.notifier.send(message, AlertLevel.STATS)
    
    def _calculate_win_rate(self) -> float:
        if len(self.stats.trades) < 10:
            return 0.5
        
        recent_trades = list(self.stats.trades)[-20:]
        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        return wins / len(recent_trades) if recent_trades else 0.5
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"📊 {self.pair} - STRATEJİ ÖZETI (1-MIN + CVD)")
        print(f"{'='*60}")
        print(f"Closed bars: {len(self.bar_builder.get_closed_bars())}")
        print(f"Cumulative CVD: {self.cvd.get_cumulative_cvd():+.2f}")
        self.stats.print_report()


class HighSharpeStrategyWithCVD:
    """Multi-pair strategy with 1-minute bars + CVD"""
    
    def __init__(
        self,
        pairs: List[str],
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        stats_report_interval: int = 3600
    ):
        self.pairs = pairs
        
        self.notifier: Optional[TelegramNotifier] = None
        if telegram_token and telegram_chat_id:
            self.notifier = TelegramNotifier(
                bot_token=telegram_token,
                chat_id=telegram_chat_id,
                enabled=True
            )
        
        self.strategies: Dict[str, PairStrategyWithCVD] = {}
        for pair in pairs:
            self.strategies[pair] = PairStrategyWithCVD(
                pair=pair,
                notifier=self.notifier,
                stats_report_interval=stats_report_interval
            )
        
        print(f"\n✅ Strategy başlatıldı - 1-MINUTE BARS + CVD (VOLUME MOMENTUM)")
        print(f"   Pairs: {', '.join(pairs)}")
        print(f"   CVD ile daha karlı stratejiye hoş geldiniz!")
    
    async def on_trade(self, pair: str, data: Dict):
        if pair not in self.strategies:
            return
        
        strategy = self.strategies[pair]
        await strategy.on_trade(
            price=data['price'],
            side=data['side'],
            amount=data['amount'],
            timestamp=data['timestamp']
        )
    
    def get_all_stats(self) -> Dict[str, Dict]:
        stats = {}
        for pair, strategy in self.strategies.items():
            stats[pair] = strategy.stats.get_summary()
        return stats
    
    def print_all_summaries(self):
        print(f"\n{'='*60}")
        print(f"📊 TÜÜN PAIR'LER - İSTATİSTİKLER (CVD STRATEGY)")
        print(f"{'='*60}")
        
        for pair, strategy in self.strategies.items():
            strategy.print_summary()
            print()


async def main():
    import os
    telegram_token = ""
    telegram_chat_id = ""

    # ✅ DÜZELTILMIŞ: Daha liquid pairler (yüksek volume)
    # Tier 1: En yüksek volume (önerilir)
    # pairs = ["BTCTRY", "ETHTRY", "SOLTRY", "AVXTRY"]

    # Tier 2: Orta volume
    # pairs = ["USDTTRY", "LINKTRY", "DOGETRY"]

    # Test için (düşük volume - DİKKAT!)
    pairs = ["UNITRY", "CVCTRY", "SPELLTRY"]

    # Strategy
    strategy = HighSharpeStrategyWithCVD(
        pairs=pairs,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        stats_report_interval=3600
    )

    # Feed
    feed = BTCTurkTradeFeed(
        pairs=pairs,
        on_trade=strategy.on_trade
    )

    print(f"\n{'='*80}")
    print(f"🚀 DÜZELTILMIŞ STRATEGY - 1-MIN BARS + CVD + TRAILING STOP")
    print(f"{'='*80}")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Period: 1 minute (OHLCV)")
    print(f"")
    print(f"✅ ÖNEMLİ DÜZELTMELER:")
    print(f"  • Sabit SL/TP (entry anında hesaplanıp SABİTLENİR)")
    print(f"  • NET PnL hesaplama (Fee + Slippage dahil)")
    print(f"  • Trailing Stop (2×ATR kar sonra aktif)")
    print(f"  • LONG + SHORT pozisyon desteği")
    print(f"  • Minimum R/R kontrolü (1:1.5)")
    print(f"  • Fee kontrolü (beklenen kar > fee)")
    print(f"")
    print(f"Indicators: ATR + RSI + SMA + CVD")
    print(f"SL/TP: 2×ATR / 3×ATR (R/R = 1:1.5)")
    print(f"Fees: 0.16% (taker) + 0.05% (slippage) = ~0.42% total")
    print(f"Telegram: {'✅ Enabled' if strategy.notifier else '❌ Disabled'}")
    print(f"{'='*80}\n")

    if strategy.notifier:
        await strategy.notifier.send(
            f"<b>🚀 DÜZELTILMIŞ Strategy Started</b>\n\n"
            f"<b>Pairs:</b> {', '.join(strategy.pairs)}\n"
            f"<b>Features:</b> Sabit SL/TP, NET PnL, Trailing Stop, LONG/SHORT\n"
            f"<b>R/R:</b> 1:1.5 | <b>Fees:</b> ~0.42%",
            AlertLevel.STATS
        )

    feed_task = asyncio.create_task(feed.start())

    try:
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        print("\n⛔ Stopping...")
        strategy.print_all_summaries()
        await feed.stop()


if __name__ == "__main__":
    asyncio.run(main())
