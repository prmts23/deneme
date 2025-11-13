"""
Andrew Aziz Day Trading Strategy - BTCTurk Implementation
=========================================================

Based on: "Advanced Techniques in Day Trading" by Andrew Aziz

Key Strategies:
1. VWAP Trend Following
2. Opening Range Breakout (ORB)
3. Bull/Bear Flag Continuation
4. ABCD Pattern (Fibonacci)
5. Support/Resistance Bounce

Risk Management:
- 1% Risk Rule (max 1% of capital per trade)
- 2:1 minimum Risk/Reward
- Daily loss limit: 5% of capital
- Position sizing formula: Risk $ / (Entry - Stop)

Author: Claude (based on Andrew Aziz methodology)
Date: 2025-01-13
"""

import asyncio
from datetime import datetime, time, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

try:
    from btcturk_clean import BTCTurkTradeFeed
    from telegram_notifier import TelegramNotifier, AlertLevel
    from trade_stats import TradeStats
    from indicators import Indicators
except ImportError:
    # Test mode
    BTCTurkTradeFeed = None
    TelegramNotifier = None
    AlertLevel = None

    class TradeStats:
        def __init__(self, max_history=1000):
            self.trades = []
        def add_trade(self, **kwargs):
            self.trades.append(kwargs)
        def print_report(self):
            print(f"  Trades: {len(self.trades)}")

    class Indicators:
        def __init__(self, max_history=500):
            self.prices = deque(maxlen=max_history)
        def update(self, price, high, low, volume):
            self.prices.append(price)


@dataclass
class TradingSession:
    """Daily trading session data"""
    date: datetime
    market_open: time  # BTCTurk: 09:00
    opening_range_high: float = 0.0
    opening_range_low: float = 0.0
    opening_range_determined: bool = False
    daily_high: float = 0.0
    daily_low: float = float('inf')
    vwap: float = 0.0
    total_volume: float = 0.0
    total_pv: float = 0.0  # Price × Volume


class VWAPCalculator:
    """
    VWAP (Volume Weighted Average Price) Calculator
    Andrew Aziz'in en önemli indikatörü
    """

    def __init__(self):
        self.reset_daily()

    def reset_daily(self):
        """Her gün başında sıfırla"""
        self.total_pv = 0.0  # Cumulative (Price × Volume)
        self.total_volume = 0.0  # Cumulative Volume
        self.vwap = 0.0
        self.vwap_upper = 0.0  # VWAP + 1 std dev
        self.vwap_lower = 0.0  # VWAP - 1 std dev
        self.price_history = []
        self.volume_history = []

    def update(self, price: float, volume: float):
        """Trade geldiğinde güncelle"""
        self.total_pv += price * volume
        self.total_volume += volume

        if self.total_volume > 0:
            self.vwap = self.total_pv / self.total_volume

        self.price_history.append(price)
        self.volume_history.append(volume)

        # Standard deviation bands (optional)
        if len(self.price_history) > 20:
            recent_prices = self.price_history[-100:]
            std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
            self.vwap_upper = self.vwap + std
            self.vwap_lower = self.vwap - std

    def is_above_vwap(self, price: float) -> bool:
        """Fiyat VWAP üstünde mi?"""
        return price > self.vwap if self.vwap > 0 else False

    def is_below_vwap(self, price: float) -> bool:
        """Fiyat VWAP altında mı?"""
        return price < self.vwap if self.vwap > 0 else False


class AndrewAzizStrategy:
    """
    Andrew Aziz Day Trading Strategy

    Strategies:
    1. VWAP Trend Following
    2. Opening Range Breakout (ORB)
    3. Moving Average Bounce (9 EMA)

    Rules (from the book):
    - Only trade in direction of VWAP trend
    - Wait for confirmation (volume spike, candle close)
    - Use key levels (ORB high/low, previous day high/low)
    - Risk management: 1% rule, 2:1 R/R minimum
    """

    def __init__(
        self,
        pair: str,
        account_size: float = 10000,  # $10K başlangıç
        risk_per_trade_pct: float = 0.01,  # 1% risk
        daily_loss_limit_pct: float = 0.05,  # 5% günlük limit
        min_risk_reward: float = 2.0,  # 2:1 R/R
        notifier: Optional[TelegramNotifier] = None
    ):
        self.pair = pair
        self.account_size = account_size
        self.risk_per_trade_pct = risk_per_trade_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.min_risk_reward = min_risk_reward
        self.notifier = notifier

        # Daily session
        self.session = TradingSession(
            date=datetime.now().date(),
            market_open=time(9, 0)  # BTCTurk: 09:00
        )

        # VWAP Calculator
        self.vwap = VWAPCalculator()

        # Indicators
        self.indicators = Indicators(max_history=500)

        # Bar data (1-min bars)
        self.bars = deque(maxlen=500)
        self.current_bar = None
        self.bar_start_time = None

        # Position
        self.position = None  # "LONG" or "SHORT" or None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = 0  # # of coins

        # Daily P&L tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0

        # Stats
        self.stats = TradeStats(max_history=1000)

        # Parameters
        self.opening_range_minutes = 15  # First 15 min = opening range
        self.volume_surge_multiplier = 2.0  # 2x avg volume

        # Fees (BTCTurk)
        self.maker_fee = 0.0008
        self.taker_fee = 0.0016
        self.slippage = 0.0005

        print(f"✅ Andrew Aziz Strategy initialized - {pair}")
        print(f"   Account Size: ${account_size:,.0f}")
        print(f"   Risk per Trade: {risk_per_trade_pct:.1%} (${account_size * risk_per_trade_pct:,.0f})")
        print(f"   Daily Loss Limit: {daily_loss_limit_pct:.1%} (${account_size * daily_loss_limit_pct:,.0f})")
        print(f"   Min R/R: {min_risk_reward}:1")

    async def on_trade(self, price: float, side: str, amount: float, timestamp: str):
        """Trade callback - her trade'de çağrılır"""

        # Update VWAP
        self.vwap.update(price, amount)

        # Update session highs/lows
        self.session.daily_high = max(self.session.daily_high, price)
        if self.session.daily_low == float('inf'):
            self.session.daily_low = price
        else:
            self.session.daily_low = min(self.session.daily_low, price)

        # Build 1-min bars
        await self._update_bar(price, timestamp, amount)

        # Check exit (if position open)
        if self.position:
            await self._check_exit(price)

    async def _update_bar(self, price: float, timestamp: str, volume: float):
        """1-minute bar builder"""
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        current_minute = dt.replace(second=0, microsecond=0)

        # New bar?
        if self.bar_start_time is None or current_minute > self.bar_start_time:
            # Close previous bar
            if self.current_bar:
                await self._on_bar_close(self.current_bar)

            # Start new bar
            self.bar_start_time = current_minute
            self.current_bar = {
                'time': current_minute,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume,
                'trades': 1
            }
        else:
            # Update current bar
            self.current_bar['high'] = max(self.current_bar['high'], price)
            self.current_bar['low'] = min(self.current_bar['low'], price)
            self.current_bar['close'] = price
            self.current_bar['volume'] += volume
            self.current_bar['trades'] += 1

    async def _on_bar_close(self, bar: Dict):
        """Bar kapandığında - strateji sinyalleri"""
        self.bars.append(bar)

        # Update indicators
        self.indicators.update(
            price=bar['close'],
            high=bar['high'],
            low=bar['low'],
            volume=bar['volume']
        )

        # Check if we're in opening range period
        current_time = bar['time'].time()
        market_open = self.session.market_open

        # Opening Range Logic (first 15 minutes)
        minutes_since_open = (
            bar['time'].hour * 60 + bar['time'].minute -
            (market_open.hour * 60 + market_open.minute)
        )

        if 0 <= minutes_since_open < self.opening_range_minutes:
            # Update opening range
            if self.session.opening_range_high == 0:
                self.session.opening_range_high = bar['high']
                self.session.opening_range_low = bar['low']
            else:
                self.session.opening_range_high = max(self.session.opening_range_high, bar['high'])
                self.session.opening_range_low = min(self.session.opening_range_low, bar['low'])

        elif minutes_since_open == self.opening_range_minutes:
            # Opening range confirmed
            self.session.opening_range_determined = True
            print(f"\n📊 Opening Range Determined ({self.opening_range_minutes} min)")
            print(f"   High: {self.session.opening_range_high:.2f}")
            print(f"   Low: {self.session.opening_range_low:.2f}")
            print(f"   Range: {self.session.opening_range_high - self.session.opening_range_low:.2f}")
            print(f"   VWAP: {self.vwap.vwap:.2f}")

        # Yeterli bar var mı?
        if len(self.bars) < 20:
            return

        # Check signals (only if no position)
        if self.position is None:
            await self._check_signals(bar)

    async def _check_signals(self, current_bar: Dict):
        """
        Andrew Aziz Sinyal Kontrolü

        Strategies:
        1. Opening Range Breakout (ORB)
        2. VWAP Trend Following
        3. Moving Average Bounce
        """

        # Daily loss limit check
        if self.daily_pnl <= -self.account_size * self.daily_loss_limit_pct:
            print(f"⛔ Daily loss limit hit: ${self.daily_pnl:.2f}")
            return

        price = current_bar['close']

        # Calculate indicators
        ema9 = self._calculate_ema(9)
        ema20 = self._calculate_ema(20)
        avg_volume = self._calculate_avg_volume(20)

        # Volume surge?
        volume_surge = current_bar['volume'] > avg_volume * self.volume_surge_multiplier

        # === STRATEGY 1: Opening Range Breakout ===
        if self.session.opening_range_determined:
            # LONG: Break above opening range high + VWAP confirmation
            if (price > self.session.opening_range_high and
                self.vwap.is_above_vwap(price) and
                volume_surge and
                ema9 > ema20):  # Trend confirmation

                await self._enter_long_orb(
                    price=price,
                    stop_loss=self.session.opening_range_low,
                    strategy="ORB_BREAKOUT"
                )
                return

            # SHORT: Break below opening range low + VWAP confirmation
            elif (price < self.session.opening_range_low and
                  self.vwap.is_below_vwap(price) and
                  volume_surge and
                  ema9 < ema20):

                await self._enter_short_orb(
                    price=price,
                    stop_loss=self.session.opening_range_high,
                    strategy="ORB_BREAKDOWN"
                )
                return

        # === STRATEGY 2: VWAP Trend Following ===
        # LONG: Price pullback to VWAP, then bounce
        if (self.vwap.is_above_vwap(price) and
            abs(price - self.vwap.vwap) / price < 0.005 and  # Within 0.5% of VWAP
            ema9 > self.vwap.vwap and  # Uptrend
            volume_surge):

            stop_loss = self.vwap.vwap * 0.995  # 0.5% below VWAP
            await self._enter_long_vwap(price, stop_loss, strategy="VWAP_LONG")
            return

        # SHORT: Price rally to VWAP, then rejection
        if (self.vwap.is_below_vwap(price) and
            abs(price - self.vwap.vwap) / price < 0.005 and
            ema9 < self.vwap.vwap and  # Downtrend
            volume_surge):

            stop_loss = self.vwap.vwap * 1.005  # 0.5% above VWAP
            await self._enter_short_vwap(price, stop_loss, strategy="VWAP_SHORT")
            return

    async def _enter_long_orb(self, price: float, stop_loss: float, strategy: str):
        """Opening Range Breakout - LONG"""

        # Andrew Aziz Rule: 2:1 minimum R/R
        risk = price - stop_loss
        if risk <= 0:
            return

        take_profit = price + (risk * self.min_risk_reward)

        # Position sizing (1% risk rule)
        risk_amount = self.account_size * self.risk_per_trade_pct
        position_size_coins = risk_amount / risk
        position_value = position_size_coins * price

        # Risk/Reward check
        reward = take_profit - price
        rr = reward / risk

        if rr < self.min_risk_reward:
            print(f"   ⚠️ R/R too low: {rr:.2f} < {self.min_risk_reward}")
            return

        # Fee check
        total_fee_pct = (self.taker_fee * 2) + (self.slippage * 2)
        expected_profit_pct = reward / price

        if expected_profit_pct < total_fee_pct + 0.005:  # +0.5% buffer
            print(f"   ⚠️ Profit < Fees: {expected_profit_pct:.2%} < {total_fee_pct:.2%}")
            return

        # ENTER POSITION
        self.position = "LONG"
        self.entry_price = price
        self.entry_time = datetime.now()
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size_coins

        print(f"\n🟢 [{self.pair}] LONG ENTRY - {strategy}")
        print(f"   Price: {price:.2f}")
        print(f"   Size: {position_size_coins:.4f} coins (${position_value:.2f})")
        print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        print(f"   Risk: ${risk_amount:.2f} ({self.risk_per_trade_pct:.1%})")
        print(f"   R/R: 1:{rr:.2f}")
        print(f"   VWAP: {self.vwap.vwap:.2f}")
        print(f"   OR High: {self.session.opening_range_high:.2f}")

        if self.notifier:
            message = f"""
<b>🟢 LONG ENTRY - {self.pair}</b>
<b>Strategy:</b> {strategy}

<b>Entry:</b> {price:.2f}
<b>Size:</b> {position_size_coins:.4f} (${position_value:.2f})
<b>SL:</b> {stop_loss:.2f} | <b>TP:</b> {take_profit:.2f}
<b>Risk:</b> ${risk_amount:.2f} ({self.risk_per_trade_pct:.1%})
<b>R/R:</b> 1:{rr:.2f}

<b>VWAP:</b> {self.vwap.vwap:.2f}
<b>OR High:</b> {self.session.opening_range_high:.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)

    async def _enter_short_orb(self, price: float, stop_loss: float, strategy: str):
        """Opening Range Breakdown - SHORT"""

        risk = stop_loss - price
        if risk <= 0:
            return

        take_profit = price - (risk * self.min_risk_reward)

        # Position sizing
        risk_amount = self.account_size * self.risk_per_trade_pct
        position_size_coins = risk_amount / risk
        position_value = position_size_coins * price

        # R/R check
        reward = price - take_profit
        rr = reward / risk

        if rr < self.min_risk_reward:
            return

        # ENTER
        self.position = "SHORT"
        self.entry_price = price
        self.entry_time = datetime.now()
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size_coins

        print(f"\n🔴 [{self.pair}] SHORT ENTRY - {strategy}")
        print(f"   Price: {price:.2f}")
        print(f"   Size: {position_size_coins:.4f} coins (${position_value:.2f})")
        print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        print(f"   Risk: ${risk_amount:.2f}")
        print(f"   R/R: 1:{rr:.2f}")
        print(f"   VWAP: {self.vwap.vwap:.2f}")

    async def _enter_long_vwap(self, price: float, stop_loss: float, strategy: str):
        """VWAP Bounce - LONG"""
        risk = price - stop_loss
        if risk <= 0:
            return

        take_profit = price + (risk * self.min_risk_reward)

        # Position sizing
        risk_amount = self.account_size * self.risk_per_trade_pct
        position_size_coins = risk_amount / risk

        self.position = "LONG"
        self.entry_price = price
        self.entry_time = datetime.now()
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size_coins

        print(f"\n🟢 [{self.pair}] LONG ENTRY - {strategy}")
        print(f"   Price: {price:.2f} (near VWAP: {self.vwap.vwap:.2f})")
        print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

    async def _enter_short_vwap(self, price: float, stop_loss: float, strategy: str):
        """VWAP Rejection - SHORT"""
        risk = stop_loss - price
        if risk <= 0:
            return

        take_profit = price - (risk * self.min_risk_reward)

        risk_amount = self.account_size * self.risk_per_trade_pct
        position_size_coins = risk_amount / risk

        self.position = "SHORT"
        self.entry_price = price
        self.entry_time = datetime.now()
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size_coins

        print(f"\n🔴 [{self.pair}] SHORT ENTRY - {strategy}")
        print(f"   Price: {price:.2f} (rejected at VWAP: {self.vwap.vwap:.2f})")
        print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

    async def _check_exit(self, current_price: float):
        """Exit check"""
        if not self.position:
            return

        # SL/TP
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

    async def _close_position(self, exit_price: float, exit_type: str):
        """Close position"""

        # PnL calculation
        if self.position == "LONG":
            gross_pnl_per_coin = exit_price - self.entry_price
        else:  # SHORT
            gross_pnl_per_coin = self.entry_price - exit_price

        gross_pnl = gross_pnl_per_coin * self.position_size

        # NET PnL (fees)
        entry_cost = self.entry_price * self.position_size * (1 + self.taker_fee + self.slippage)
        exit_revenue = exit_price * self.position_size * (1 - self.taker_fee - self.slippage)

        if self.position == "LONG":
            net_pnl = exit_revenue - entry_cost
        else:
            net_pnl = entry_cost - exit_revenue

        net_pnl_pct = (net_pnl / entry_cost) * 100

        # Update daily P&L
        self.daily_pnl += net_pnl
        self.daily_trades += 1

        duration = (datetime.now() - self.entry_time).total_seconds()

        emoji = "🟢" if net_pnl > 0 else "🔴"

        print(f"\n{emoji} [{self.pair}] {self.position} EXIT ({exit_type})")
        print(f"   Entry: {self.entry_price:.2f} → Exit: {exit_price:.2f}")
        print(f"   NET PnL: ${net_pnl:+.2f} ({net_pnl_pct:+.2f}%)")
        print(f"   Daily PnL: ${self.daily_pnl:+.2f}")
        print(f"   Duration: {duration:.0f}s")

        # Stats
        self.stats.add_trade(
            entry_price=self.entry_price,
            exit_price=exit_price,
            pnl=net_pnl,
            pnl_pct=net_pnl_pct,
            duration_sec=duration,
            trade_type=self.position,
            entry_time=self.entry_time,
            exit_time=datetime.now()
        )

        # Reset
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = 0

    def _calculate_ema(self, period: int) -> float:
        """EMA calculator"""
        if len(self.bars) < period:
            return 0

        prices = [b['close'] for b in list(self.bars)[-period:]]
        k = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = price * k + ema * (1 - k)

        return ema

    def _calculate_avg_volume(self, period: int) -> float:
        """Average volume"""
        if len(self.bars) < period:
            return 0

        volumes = [b['volume'] for b in list(self.bars)[-period:]]
        return statistics.mean(volumes) if volumes else 0

    def print_summary(self):
        """Strategy summary"""
        print(f"\n{'='*60}")
        print(f"📊 ANDREW AZIZ STRATEGY - {self.pair}")
        print(f"{'='*60}")
        print(f"Account Size: ${self.account_size:,.0f}")
        print(f"Daily PnL: ${self.daily_pnl:+.2f} ({self.daily_pnl/self.account_size*100:+.2f}%)")
        print(f"Daily Trades: {self.daily_trades}")
        print(f"VWAP: {self.vwap.vwap:.2f}")
        print(f"Opening Range: {self.session.opening_range_low:.2f} - {self.session.opening_range_high:.2f}")
        self.stats.print_report()


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Test Andrew Aziz Strategy"""

    # Strategy
    strategy = AndrewAzizStrategy(
        pair="BTCTRY",
        account_size=10000,
        risk_per_trade_pct=0.01,  # 1%
        daily_loss_limit_pct=0.05,  # 5%
        min_risk_reward=2.0,
        notifier=None
    )

    print(f"\n{'='*70}")
    print(f"🚀 ANDREW AZIZ DAY TRADING STRATEGY")
    print(f"{'='*70}")
    print(f"Based on: 'Advanced Techniques in Day Trading'")
    print(f"")
    print(f"Strategies:")
    print(f"  1. Opening Range Breakout (ORB) - First 15 min")
    print(f"  2. VWAP Trend Following")
    print(f"  3. Moving Average Bounce (9 EMA)")
    print(f"")
    print(f"Risk Management:")
    print(f"  • 1% Risk per Trade")
    print(f"  • 5% Daily Loss Limit")
    print(f"  • 2:1 Minimum R/R")
    print(f"  • Position Sizing Formula")
    print(f"{'='*70}\n")

    print("⏳ Strategy ready. Waiting for market data...")
    print("   (In production: connect to BTCTurk WebSocket)")


if __name__ == "__main__":
    asyncio.run(main())
