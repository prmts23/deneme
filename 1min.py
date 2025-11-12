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
    """1-minute bars + CVD ile strateji"""
    
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
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_bar_index = None
        self.entry_cvd_trend = None
        
        # Stats
        self.stats = TradeStats(max_history=1000)
        self.last_stats_report = datetime.now()
        
        # Parameters
        self.atr_period = 14
        self.rsi_period = 14
        self.win_rate_threshold = 0.49
        
        # CVD Parameters
        self.cvd_momentum_period = 5
        self.cvd_delta_threshold = 0.5  # Min. delta ratio
        
        print(f"✅ {pair} strategy başlatıldı (1-MIN + CVD)")
    
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
        """Açık pozisyon kontrol"""
        if self.position is None:
            return
        
        atr = self.indicators.calculate_atr(self.atr_period)
        if atr == 0:
            return
        k1 = 2.0  # SL katsayısı
        min_edge = 0.0036  # ≈ %0.36 (fee+spread+slip+güvenlik)

        required_k2 = k1 + (current_price * min_edge) / max(atr, 1e-9)
        k2 = max(3.8, required_k2)  # 3.8×ATR taban, gerekirse yükselt

        stop_loss   = self.entry_price - k1 * atr
        take_profit = self.entry_price + k2 * atr

        
        if current_price >= take_profit:
            await self._close_position(current_price, "TAKE PROFIT")
        elif current_price <= stop_loss:
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
        """Sinyal kontrolü - CVD ile güçlendirilmiş"""
        
        if win_rate < self.win_rate_threshold:
            return
        
        # LONG Signal (CVD ile filtreleme)
        if self.position is None:
            # RSI + Price + CVD
            # CVD BULLISH (buy volume > sell volume)
            # CVD momentum pozitif (alıcılar kontrol ediyor)
            # Buy ratio > 0.55 (alıcılar baskın)
            
            rsi_ok = rsi < 50                 # önce 35’ti
            price_ok = current_price > sma20  # önce > SMA50 ve SMA20'e 0.5*ATR yakın
            cvd_ok = (
                cvd_momentum >= 0 and         # >0 yerine >=0
                buy_ratio > 0.52 and          # 0.55 → 0.52
                cvd_trend in ["BULLISH", "NEUTRAL"]
            )

            if atr > 0 and rsi_ok and price_ok and cvd_ok:
                self.position = "LONG"
                self.entry_price = current_price
                self.entry_time = datetime.now()
                self.entry_bar_index = len(self.indicators.prices)
                self.entry_cvd_trend = cvd_trend
                
                stop_loss = current_price - atr * 2
                take_profit = current_price + atr * 3
                risk = current_price - stop_loss
                reward = take_profit - current_price
                risk_reward = reward / risk if risk > 0 else 0
                
                print(f"\n🟢 [{self.pair}] LONG ENTRY @ {current_price:.2f}")
                print(f"   ATR: {atr:.4f} | RSI: {rsi:.1f}")
                print(f"   CVD Trend: {cvd_trend} | Momentum: {cvd_momentum:+.2f} | Buy: {buy_ratio:.1%}")
                print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                print(f"   R:R: 1:{risk_reward:.2f}")
                
                if self.notifier:
                    message = f"""
<b>🟢 LONG ENTRY - {self.pair}</b>

<b>Price:</b> {current_price:.2f}
<b>Stop Loss:</b> {stop_loss:.2f}
<b>Take Profit:</b> {take_profit:.2f}
<b>Risk/Reward:</b> 1:{risk_reward:.2f}

<b>Indicators:</b>
├─ RSI: {rsi:.1f}
├─ ATR: {atr:.4f}
├─ SMA20: {sma20:.2f}
└─ SMA50: {sma50:.2f}

<b>Volume Indicators:</b>
├─ CVD Trend: {cvd_trend}
├─ CVD Momentum: {cvd_momentum:+.2f}
├─ Buy Ratio: {buy_ratio:.1%}
└─ CVD Delta: {cvd_delta:+.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
                    await self.notifier.send(message, AlertLevel.TRADE)
    
    async def _close_position(self, exit_price: float, exit_type: str):
        """Pozisyonu kapat"""
        pnl = exit_price - self.entry_price
        pnl_pct = (pnl / self.entry_price) * 100
        duration = (datetime.now() - self.entry_time).total_seconds()
        
        emoji = "🟢" if pnl > 0 else "🔴"
        bars_held = len(self.indicators.prices) - self.entry_bar_index
        
        print(f"\n{emoji} [{self.pair}] LONG EXIT ({exit_type}) @ {exit_price:.2f}")
        print(f"   PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"   Bars: {bars_held} | Duration: {duration:.0f}s")
        
        self.stats.add_trade(
            entry_price=self.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_sec=duration,
            trade_type="LONG",
            entry_time=self.entry_time,
            exit_time=datetime.now()
        )
        
        if self.notifier:
            message = f"""
<b>{emoji} LONG EXIT - {self.pair}</b>

<b>Entry:</b> {self.entry_price:.2f}
<b>Exit:</b> {exit_price:.2f}
<b>PnL:</b> {pnl:+.2f} ({pnl_pct:+.2f}%)
<b>Duration:</b> {duration:.0f}s ({bars_held} bars)
<b>Exit Type:</b> {exit_type}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)
        
        self.position = None
    
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
    
    # Strategy
    strategy = HighSharpeStrategyWithCVD(
        pairs=["UNITRY", "CVCTRY","SPELLTRY"],
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        stats_report_interval=360
    )
    
    # Feed
    feed = BTCTurkTradeFeed(
        pairs=["UNITRY", "CVCTRY","SPELLTRY"],
        on_trade=strategy.on_trade
    )
    
    print(f"\n{'='*70}")
    print(f"🚀 HIGH SHARPE STRATEGY - 1-MIN BARS + CVD VOLUME MOMENTUM")
    print(f"{'='*70}")
    print(f"Pairs: BTCTRY, ETHTRY")
    print(f"Period: 1 minute (OHLCV)")
    print(f"Volume Indicator: CVD (Cumulative Volume Delta)")
    print(f"Indicators: ✅ ATR + RSI + SMA + CVD")
    print(f"Features:")
    print(f"  • Price action (RSI, ATR, SMA)")
    print(f"  • Volume momentum (CVD, Buy Ratio)")
    print(f"  • Multi-pair support")
    print(f"  • Telegram alerts")
    print(f"Telegram: {'✅ Enabled' if strategy.notifier else '❌ Disabled'}")
    await strategy.notifier.send(
    f"<b>🚀 Strategy started</b>\nPairs: {', '.join(strategy.pairs)}",AlertLevel.STATS)
    print(f"{'='*70}\n")
    
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
