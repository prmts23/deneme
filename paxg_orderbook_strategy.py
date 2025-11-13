"""
PAXGUSDT Order Book Strategy - Market Making + Imbalance Trading
=================================================================

PAXG (altın token) için özel order book tabanlı strateji:
- Spread capture (market making)
- Order book imbalance detection
- Inventory risk management
- Low volatility scalping
- Binance PAXGUSDT lead-lag (opsiyonel)

Author: Claude (Algo Trading Expert)
Date: 2025-01-13
"""

import asyncio
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import statistics

try:
    from btcturk_clean import BTCTurkTradeFeed
    from telegram_notifier import TelegramNotifier, AlertLevel
    from trade_stats import TradeStats
except ImportError:
    # Test mode - mock classes
    BTCTurkTradeFeed = None
    TelegramNotifier = None
    AlertLevel = None

    class TradeStats:
        def __init__(self, max_history=1000):
            self.trades = []
        def add_trade(self, **kwargs):
            self.trades.append(kwargs)
        def print_report(self):
            print(f"  Trades recorded: {len(self.trades)}")


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    spread_pct: float


class OrderBookManager:
    """
    Order book yönetimi - L2 data
    BTCTurk WebSocket'ten order book snapshot'ları alır
    """

    def __init__(self, pair: str, depth: int = 20):
        self.pair = pair
        self.depth = depth

        # Current order book
        self.bids: List[Tuple[float, float]] = []  # [(price, size), ...]
        self.asks: List[Tuple[float, float]] = []

        # History
        self.snapshots = deque(maxlen=100)

        # Metrics
        self.last_update = None
        self.update_count = 0

    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Order book güncelle"""
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)[:self.depth]
        self.asks = sorted(asks, key=lambda x: x[0])[:self.depth]

        self.last_update = datetime.now()
        self.update_count += 1

        # Snapshot kaydet
        snapshot = self.get_snapshot()
        if snapshot:
            self.snapshots.append(snapshot)

    def get_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Mevcut snapshot"""
        if not self.bids or not self.asks:
            return None

        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = spread / mid

        return OrderBookSnapshot(
            timestamp=datetime.now(),
            bids=self.bids.copy(),
            asks=self.asks.copy(),
            mid_price=mid,
            spread=spread,
            spread_pct=spread_pct
        )

    def calculate_imbalance(self, levels: int = 5) -> float:
        """
        Order book imbalance (top N levels)

        Returns:
            -1 to +1: negative = sell pressure, positive = buy pressure
        """
        if not self.bids or not self.asks:
            return 0.0

        bid_volume = sum(size for price, size in self.bids[:levels])
        ask_volume = sum(size for price, size in self.asks[:levels])

        if bid_volume + ask_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / (bid_volume + ask_volume)

    def calculate_vwap_bid_ask(self, levels: int = 5) -> Tuple[float, float]:
        """
        Volume-weighted average price (bid ve ask için)
        """
        if not self.bids or not self.asks:
            return 0.0, 0.0

        # VWAP Bid (alıcı tarafı)
        bid_sum_pv = sum(price * size for price, size in self.bids[:levels])
        bid_sum_v = sum(size for price, size in self.bids[:levels])
        vwap_bid = bid_sum_pv / bid_sum_v if bid_sum_v > 0 else 0.0

        # VWAP Ask (satıcı tarafı)
        ask_sum_pv = sum(price * size for price, size in self.asks[:levels])
        ask_sum_v = sum(size for price, size in self.asks[:levels])
        vwap_ask = ask_sum_pv / ask_sum_v if ask_sum_v > 0 else 0.0

        return vwap_bid, vwap_ask

    def get_depth_at_distance(self, distance_pct: float) -> Tuple[float, float]:
        """
        Belirli % mesafedeki toplam derinlik

        Args:
            distance_pct: 0.001 = %0.1

        Returns:
            (bid_depth, ask_depth)
        """
        if not self.bids or not self.asks:
            return 0.0, 0.0

        mid = (self.bids[0][0] + self.asks[0][0]) / 2

        # Bid depth
        bid_threshold = mid * (1 - distance_pct)
        bid_depth = sum(size for price, size in self.bids if price >= bid_threshold)

        # Ask depth
        ask_threshold = mid * (1 + distance_pct)
        ask_depth = sum(size for price, size in self.asks if price <= ask_threshold)

        return bid_depth, ask_depth


class PAXGOrderBookStrategy:
    """
    PAXGUSDT Order Book Market Making Strategy

    Strateji mantığı:
    1. Spread > threshold ise market making (bid-ask arası koy)
    2. Order book imbalance > threshold ise directional trade
    3. Inventory risk: max PAXG pozisyonu kontrol et
    4. Quick scalp: küçük kar marjları (0.1-0.3%)
    """

    def __init__(
        self,
        pair: str = "PAXGUSDT",
        notifier: Optional[TelegramNotifier] = None,
        max_inventory_usd: float = 10000,  # Max $10K PAXG
        min_spread_pct: float = 0.0015,    # Min %0.15 spread
        target_profit_pct: float = 0.002,  # %0.2 target profit
        imbalance_threshold: float = 0.25   # %25 imbalance
    ):
        self.pair = pair
        self.notifier = notifier
        self.max_inventory_usd = max_inventory_usd
        self.min_spread_pct = min_spread_pct
        self.target_profit_pct = target_profit_pct
        self.imbalance_threshold = imbalance_threshold

        # Order book manager
        self.order_book = OrderBookManager(pair=pair, depth=20)

        # Position tracking
        self.position = None  # "LONG" or "SHORT" or None
        self.entry_price = None
        self.entry_time = None
        self.position_size = 0  # PAXG amount
        self.inventory_usd = 0  # Current inventory value

        # SL/TP
        self.stop_loss = None
        self.take_profit = None

        # Stats
        self.stats = TradeStats(max_history=1000)
        self.total_spreads_captured = 0.0
        self.total_trades = 0

        # Fees (BTCTurk)
        self.maker_fee = 0.0008  # 0.08%
        self.taker_fee = 0.0016  # 0.16%
        self.slippage = 0.0003   # 0.03% (PAXG düşük volatilite)

        # Market making parameters
        self.mm_size_usd = 500  # Her market making trade $500
        self.mm_placement_offset = 0.0005  # Bid/ask'dan 0.05% içeride

        # Imbalance trading parameters
        self.imb_size_usd = 1000  # Imbalance trade $1000
        self.imb_quick_exit_pct = 0.0015  # %0.15 kar al hızla çık

        print(f"✅ PAXG Order Book Strategy initialized")
        print(f"   Pair: {pair}")
        print(f"   Max Inventory: ${max_inventory_usd:,.0f}")
        print(f"   Min Spread: {min_spread_pct:.2%}")
        print(f"   Target Profit: {target_profit_pct:.2%}")
        print(f"   Imbalance Threshold: {imbalance_threshold:.1%}")

    def update_order_book(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Order book güncelle"""
        self.order_book.update(bids, asks)

    async def on_order_book_update(self):
        """
        Order book her güncellendiğinde çağrılır
        Ana strateji loop
        """
        snapshot = self.order_book.get_snapshot()
        if not snapshot:
            return

        # 1. Açık pozisyon var mı? Çıkış kontrolü
        if self.position:
            await self._check_exit(snapshot)

        # 2. Yeni fırsat ara
        else:
            # 2a. Market making fırsatı
            if await self._check_market_making_opportunity(snapshot):
                await self._execute_market_making(snapshot)

            # 2b. Imbalance trading fırsatı
            elif await self._check_imbalance_opportunity(snapshot):
                await self._execute_imbalance_trade(snapshot)

    async def _check_market_making_opportunity(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Market making fırsatı var mı?

        Koşullar:
        1. Spread > min_spread_pct (yeterli kar marjı)
        2. Inventory < max (risk kontrolü)
        3. Order book derinliği yeterli
        """
        # Spread yeterli mi?
        if snapshot.spread_pct < self.min_spread_pct:
            return False

        # Inventory risk
        if abs(self.inventory_usd) > self.max_inventory_usd * 0.8:
            print(f"   ⚠️ Inventory too high: ${self.inventory_usd:,.0f}")
            return False

        # Depth check (top 5 levels)
        bid_depth, ask_depth = self.order_book.get_depth_at_distance(0.002)  # %0.2
        min_depth_usd = self.mm_size_usd * 3  # En az 3× trade size

        if bid_depth * snapshot.mid_price < min_depth_usd:
            return False
        if ask_depth * snapshot.mid_price < min_depth_usd:
            return False

        return True

    async def _execute_market_making(self, snapshot: OrderBookSnapshot):
        """
        Market making: bid-ask arasına limit order koy

        Strateji:
        - Bid'e yakın alış, ask'e yakın satış limit order
        - Spread capture
        """
        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        mid = snapshot.mid_price

        # Placement prices (spread içinde, biraz daha iyi fiyat)
        buy_price = best_bid + (snapshot.spread * self.mm_placement_offset)
        sell_price = best_ask - (snapshot.spread * self.mm_placement_offset)

        # Expected profit
        expected_profit_pct = (sell_price - buy_price) / buy_price

        # Fee kontrolü
        total_fee = (self.maker_fee * 2) + (self.slippage * 2)
        if expected_profit_pct < total_fee + 0.0005:  # +0.05% buffer
            print(f"   ⚠️ MM profit too low: {expected_profit_pct:.2%} < {total_fee:.2%}")
            return

        # Position size (USD bazlı)
        position_size_paxg = self.mm_size_usd / mid

        print(f"\n🎯 MARKET MAKING OPPORTUNITY")
        print(f"   Spread: {snapshot.spread_pct:.3%}")
        print(f"   Buy: {buy_price:.2f} | Sell: {sell_price:.2f}")
        print(f"   Expected: {expected_profit_pct:.3%} (fee: {total_fee:.3%})")
        print(f"   Size: {position_size_paxg:.4f} PAXG (${self.mm_size_usd})")

        # SIMÜLASYON: Gerçek order placement burada olacak
        # Bu örnekte sadece LONG entry yapıyoruz (basit)
        await self._open_position(
            side="LONG",
            entry_price=buy_price,
            size=position_size_paxg,
            strategy="MARKET_MAKING",
            snapshot=snapshot
        )

    async def _check_imbalance_opportunity(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Order book imbalance fırsatı

        Koşullar:
        1. Imbalance > threshold (güçlü alıcı/satıcı baskısı)
        2. Spread normal (< %0.5)
        3. Son 10 saniyede imbalance tutarlı
        """
        imbalance = self.order_book.calculate_imbalance(levels=5)

        # Imbalance yeterli mi?
        if abs(imbalance) < self.imbalance_threshold:
            return False

        # Spread çok geniş değil mi? (likidite iyi)
        if snapshot.spread_pct > 0.005:  # > %0.5
            return False

        # Inventory risk
        # Eğer imbalance BUY yönlü ama biz zaten LONG'sak → skip
        if imbalance > 0 and self.inventory_usd > self.max_inventory_usd * 0.5:
            return False
        if imbalance < 0 and self.inventory_usd < -self.max_inventory_usd * 0.5:
            return False

        # Son 5 snapshot'ta tutarlı mı?
        if len(self.order_book.snapshots) < 5:
            return False

        recent_imbalances = []
        for snap in list(self.order_book.snapshots)[-5:]:
            # Recalculate imbalance (basit approx)
            bid_vol = sum(s for p, s in snap.bids[:5])
            ask_vol = sum(s for p, s in snap.asks[:5])
            imb = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
            recent_imbalances.append(imb)

        avg_imbalance = statistics.mean(recent_imbalances) if recent_imbalances else 0

        # Tutarlı mı?
        if abs(avg_imbalance) < self.imbalance_threshold * 0.8:
            return False

        # Yön uyumlu mu?
        def sign(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)

        if sign(imbalance) != sign(avg_imbalance):
            return False

        return True

    async def _execute_imbalance_trade(self, snapshot: OrderBookSnapshot):
        """
        Order book imbalance'a göre trade

        Logic:
        - Imbalance > 0 (alıcı baskısı) → LONG
        - Imbalance < 0 (satıcı baskısı) → SHORT
        """
        imbalance = self.order_book.calculate_imbalance(levels=5)

        side = "LONG" if imbalance > 0 else "SHORT"
        entry_price = snapshot.asks[0][0] if side == "LONG" else snapshot.bids[0][0]

        position_size_paxg = self.imb_size_usd / snapshot.mid_price

        print(f"\n⚡ ORDER BOOK IMBALANCE SIGNAL")
        print(f"   Imbalance: {imbalance:+.2%}")
        print(f"   Side: {side}")
        print(f"   Entry: {entry_price:.2f}")
        print(f"   Size: {position_size_paxg:.4f} PAXG (${self.imb_size_usd})")

        await self._open_position(
            side=side,
            entry_price=entry_price,
            size=position_size_paxg,
            strategy="IMBALANCE",
            snapshot=snapshot
        )

    async def _open_position(
        self,
        side: str,
        entry_price: float,
        size: float,
        strategy: str,
        snapshot: OrderBookSnapshot
    ):
        """Pozisyon aç"""

        # SL/TP hesapla
        if strategy == "MARKET_MAKING":
            # Tight SL (spread kaybedilirse çık)
            sl_pct = 0.003  # %0.3
            tp_pct = self.target_profit_pct  # %0.2

        elif strategy == "IMBALANCE":
            # Quick scalp
            sl_pct = 0.002  # %0.2
            tp_pct = self.imb_quick_exit_pct  # %0.15

        else:
            sl_pct = 0.005
            tp_pct = 0.005

        if side == "LONG":
            self.stop_loss = entry_price * (1 - sl_pct)
            self.take_profit = entry_price * (1 + tp_pct)
        else:  # SHORT
            self.stop_loss = entry_price * (1 + sl_pct)
            self.take_profit = entry_price * (1 - tp_pct)

        # Position state
        self.position = side
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.position_size = size if side == "LONG" else -size
        self.inventory_usd = self.position_size * entry_price

        risk = abs(entry_price - self.stop_loss)
        reward = abs(self.take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0

        print(f"✅ {side} ENTRY @ {entry_price:.2f}")
        print(f"   Size: {abs(self.position_size):.4f} PAXG")
        print(f"   SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")
        print(f"   R:R: 1:{rr:.2f}")
        print(f"   Strategy: {strategy}")

        if self.notifier:
            message = f"""
<b>{'🟢' if side == 'LONG' else '🔴'} {side} ENTRY - {self.pair}</b>

<b>Strategy:</b> {strategy}
<b>Entry:</b> {entry_price:.2f}
<b>Size:</b> {abs(self.position_size):.4f} PAXG
<b>SL:</b> {self.stop_loss:.2f} | <b>TP:</b> {self.take_profit:.2f}
<b>R:R:</b> 1:{rr:.2f}

<b>Order Book:</b>
├─ Spread: {snapshot.spread_pct:.3%}
├─ Imbalance: {self.order_book.calculate_imbalance():.2%}
└─ Mid: {snapshot.mid_price:.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)

    async def _check_exit(self, snapshot: OrderBookSnapshot):
        """Exit kontrolü"""
        if not self.position or not self.entry_time:
            return

        current_price = snapshot.mid_price

        # SL/TP check
        if self.position == "LONG":
            if current_price >= self.take_profit:
                await self._close_position(current_price, "TAKE PROFIT", snapshot)
                return
            elif current_price <= self.stop_loss:
                await self._close_position(current_price, "STOP LOSS", snapshot)
                return

        elif self.position == "SHORT":
            if current_price <= self.take_profit:
                await self._close_position(current_price, "TAKE PROFIT", snapshot)
                return
            elif current_price >= self.stop_loss:
                await self._close_position(current_price, "STOP LOSS", snapshot)
                return

        # Time-based exit (max 5 min hold)
        duration = (datetime.now() - self.entry_time).total_seconds()
        if duration > 300:  # 5 dakika
            await self._close_position(current_price, "TIMEOUT", snapshot)

    async def _close_position(self, exit_price: float, exit_type: str, snapshot: OrderBookSnapshot):
        """Pozisyonu kapat"""

        # PnL hesapla
        if self.position == "LONG":
            gross_pnl = (exit_price - self.entry_price) * abs(self.position_size)
        else:  # SHORT
            gross_pnl = (self.entry_price - exit_price) * abs(self.position_size)

        gross_pnl_pct = (gross_pnl / (self.entry_price * abs(self.position_size))) * 100

        # NET PnL (fees)
        entry_cost = self.entry_price * abs(self.position_size) * (1 + self.taker_fee + self.slippage)
        exit_revenue = exit_price * abs(self.position_size) * (1 - self.taker_fee - self.slippage)

        if self.position == "LONG":
            net_pnl = exit_revenue - entry_cost
        else:
            net_pnl = entry_cost - exit_revenue

        net_pnl_pct = (net_pnl / entry_cost) * 100
        fee_cost = gross_pnl - net_pnl

        duration = (datetime.now() - self.entry_time).total_seconds()

        emoji = "🟢" if net_pnl > 0 else "🔴"

        print(f"\n{emoji} {self.position} EXIT ({exit_type}) @ {exit_price:.2f}")
        print(f"   Entry: {self.entry_price:.2f} | Exit: {exit_price:.2f}")
        print(f"   GROSS PnL: ${gross_pnl:+.2f} ({gross_pnl_pct:+.2f}%)")
        print(f"   NET PnL: ${net_pnl:+.2f} ({net_pnl_pct:+.2f}%)")
        print(f"   Fee: ${fee_cost:.2f}")
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

        if net_pnl > 0:
            self.total_spreads_captured += net_pnl

        self.total_trades += 1

        if self.notifier:
            message = f"""
<b>{emoji} {self.position} EXIT - {self.pair}</b>

<b>Exit Type:</b> {exit_type}
<b>Entry:</b> {self.entry_price:.2f} → <b>Exit:</b> {exit_price:.2f}

<b>PnL (GROSS):</b> ${gross_pnl:+.2f} ({gross_pnl_pct:+.2f}%)
<b>PnL (NET):</b> ${net_pnl:+.2f} ({net_pnl_pct:+.2f}%)
<b>Fee Cost:</b> ${fee_cost:.2f}

<b>Duration:</b> {duration:.0f}s
<b>Total Captured:</b> ${self.total_spreads_captured:+.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send(message, AlertLevel.TRADE)

        # Reset position
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.position_size = 0
        self.inventory_usd = 0
        self.stop_loss = None
        self.take_profit = None

    def print_summary(self):
        """Strateji özeti"""
        print(f"\n{'='*60}")
        print(f"📊 PAXG ORDER BOOK STRATEGY - SUMMARY")
        print(f"{'='*60}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Total Spreads Captured: ${self.total_spreads_captured:+.2f}")
        print(f"Current Inventory: ${self.inventory_usd:+.2f}")
        self.stats.print_report()


# ============================================================================
# SIMÜLASYON: Order Book Data Generator (Test için)
# ============================================================================

class OrderBookSimulator:
    """
    PAXGUSDT order book simülatörü (test için)
    Gerçek WebSocket yerine dummy data
    """

    def __init__(self, base_price: float = 2650.0):
        self.base_price = base_price
        self.last_price = base_price

    def generate_order_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Rastgele order book üret
        """
        # Random walk
        change = random.gauss(0, 0.0005)  # %0.05 std
        self.last_price = self.last_price * (1 + change)

        # Spread
        spread_pct = random.uniform(0.0008, 0.003)  # %0.08 - %0.3
        spread = self.last_price * spread_pct

        best_bid = self.last_price - spread / 2
        best_ask = self.last_price + spread / 2

        # Generate bids (10 levels)
        bids = []
        for i in range(10):
            price = best_bid - i * 0.5
            size = random.uniform(0.5, 5.0)  # 0.5-5 PAXG
            bids.append((price, size))

        # Generate asks (10 levels)
        asks = []
        for i in range(10):
            price = best_ask + i * 0.5
            size = random.uniform(0.5, 5.0)
            asks.append((price, size))

        # Random imbalance
        if random.random() < 0.3:  # %30 imbalance
            if random.random() < 0.5:
                # Buy pressure
                for i in range(5):
                    bids[i] = (bids[i][0], bids[i][1] * 2)
            else:
                # Sell pressure
                for i in range(5):
                    asks[i] = (asks[i][0], asks[i][1] * 2)

        return bids, asks


# ============================================================================
# MAIN - Test
# ============================================================================

async def main():
    """Test PAXG strategy"""

    # Strategy
    strategy = PAXGOrderBookStrategy(
        pair="PAXGUSDT",
        notifier=None,  # Telegram opsiyonel
        max_inventory_usd=10000,
        min_spread_pct=0.0015,
        target_profit_pct=0.002,
        imbalance_threshold=0.25
    )

    # Simulator (gerçek WebSocket yerine)
    simulator = OrderBookSimulator(base_price=2650.0)

    print(f"\n{'='*70}")
    print(f"🚀 PAXGUSDT ORDER BOOK STRATEGY - SIMULATION")
    print(f"{'='*70}")
    print(f"Strategy: Market Making + Imbalance Trading")
    print(f"Max Inventory: ${strategy.max_inventory_usd:,.0f}")
    print(f"Target: {strategy.target_profit_pct:.2%} per trade")
    print(f"Running for 60 seconds...\n")

    # Main loop (60 saniye test)
    start_time = datetime.now()
    update_count = 0

    try:
        while (datetime.now() - start_time).total_seconds() < 60:
            # Generate order book
            bids, asks = simulator.generate_order_book()

            # Update strategy
            strategy.update_order_book(bids, asks)

            # Check signals
            await strategy.on_order_book_update()

            update_count += 1

            # 100ms sleep (10 updates/saniye)
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⛔ Stopping...")

    # Summary
    print(f"\n{'='*70}")
    print(f"Simulation completed:")
    print(f"  Duration: {(datetime.now() - start_time).total_seconds():.0f}s")
    print(f"  Order Book Updates: {update_count}")
    strategy.print_summary()
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
