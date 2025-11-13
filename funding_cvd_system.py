"""
Binance Futures Funding Rate + CVD Analysis System
===================================================

Production-grade real-time analysis system for:
- Funding rate monitoring and prediction
- CVD (Cumulative Volume Delta) tracking
- Order flow analysis
- Delta neutral arbitrage signals

Features:
- WebSocket streaming (markPrice + trade)
- Real-time CVD calculation
- 1-minute bar resampling
- Feature engineering (z-scores, momentum, basis)
- Telegram notifications
- Fee and slippage accounting
- Auto-reconnect on disconnect

Author: Claude
Date: 2025-01-13
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import pandas as pd
import numpy as np
import requests
from binance.client import Client

# ====================================================================
# CONFIGURATION
# ====================================================================

# Binance API (for initial data and spot prices)
BINANCE_API_KEY = "YOUR_API_KEY"  # Not required for public endpoints
BINANCE_API_SECRET = "YOUR_SECRET"

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# Trading Parameters
SYMBOLS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]  # Symbols to monitor
POSITION_SIZE_USD = 1000  # Position size for calculations

# Fees and Slippage
SPOT_MAKER_FEE = 0.001   # 0.10%
SPOT_TAKER_FEE = 0.001   # 0.10%
FUTURES_MAKER_FEE = 0.0002  # 0.02%
FUTURES_TAKER_FEE = 0.0004  # 0.04%
SLIPPAGE = 0.0005  # 0.05% per execution

# Analysis Parameters
BUFFER_SIZE = 500  # Events before processing
RESAMPLE_INTERVAL = "1min"  # 1-minute bars
ROLLING_WINDOW_1H = 60  # 60 minutes
ROLLING_WINDOW_15M = 15  # 15 minutes

# Signal Thresholds
FUNDING_Z_THRESHOLD = 2.0  # Z-score threshold for funding spike
CVD_CHANGE_THRESHOLD = 10000  # Threshold for significant CVD change
BASIS_THRESHOLD = 0.001  # 0.1% basis threshold

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================================================================
# DATA STRUCTURES
# ====================================================================

@dataclass
class FundingData:
    """Funding rate data point"""
    timestamp: datetime
    symbol: str
    funding_rate: float
    mark_price: float
    next_funding_time: datetime

@dataclass
class TradeData:
    """Trade data point"""
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool
    signed_volume: float

@dataclass
class Signal:
    """Trading signal"""
    timestamp: datetime
    symbol: str
    signal_type: str  # "LONG", "SHORT", "CLOSE"
    confidence: float
    funding_rate: float
    cvd_change: float
    basis: float
    expected_profit_daily: float
    fees_total: float
    payback_days: float

# ====================================================================
# TELEGRAM NOTIFIER
# ====================================================================

class TelegramNotifier:
    """Send notifications via Telegram"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def send_signal(self, signal: Signal) -> bool:
        """Send trading signal notification"""
        emoji = "🟢" if signal.signal_type == "LONG" else "🔴" if signal.signal_type == "SHORT" else "⚪"

        message = f"""
{emoji} <b>{signal.signal_type} SIGNAL: {signal.symbol}</b>

📊 <b>Signal Confidence:</b> {signal.confidence:.1%}

💰 <b>Funding Rate:</b> {signal.funding_rate*100:.4f}%
📈 <b>CVD Change (15m):</b> {signal.cvd_change:,.0f}
📉 <b>Basis:</b> {signal.basis*100:.3f}%

💵 <b>Expected Daily Profit:</b> ${signal.expected_profit_daily:.2f}
💸 <b>Total Fees:</b> ${signal.fees_total:.2f}
⏱ <b>Payback Period:</b> {signal.payback_days:.1f} days

⏰ <b>Time:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)

    def send_summary(self, summary: Dict) -> bool:
        """Send periodic summary"""
        message = f"""
📊 <b>Funding Rate Summary</b>

<b>Top Opportunities:</b>
{summary.get('top_opportunities', 'None')}

<b>Market Stats:</b>
• Average Funding: {summary.get('avg_funding', 0)*100:.4f}%
• Max Funding: {summary.get('max_funding', 0)*100:.4f}%
• Active Signals: {summary.get('active_signals', 0)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)

# ====================================================================
# FEE AND SLIPPAGE CALCULATOR
# ====================================================================

class FeeCalculator:
    """Calculate fees and slippage for delta neutral positions"""

    @staticmethod
    def calculate_entry_fees(position_size: float) -> Tuple[float, Dict]:
        """
        Calculate fees for entering delta neutral position

        Entry:
        - Buy SPOT (maker if limit, taker if market)
        - Short FUTURES (maker if limit, taker if market)
        """
        # Assume market orders (taker fees) for conservative estimate
        spot_fee = position_size * SPOT_TAKER_FEE
        futures_fee = position_size * FUTURES_TAKER_FEE
        spot_slippage = position_size * SLIPPAGE
        futures_slippage = position_size * SLIPPAGE

        total_entry = spot_fee + futures_fee + spot_slippage + futures_slippage

        breakdown = {
            "spot_fee": spot_fee,
            "futures_fee": futures_fee,
            "spot_slippage": spot_slippage,
            "futures_slippage": futures_slippage,
            "total": total_entry
        }

        return total_entry, breakdown

    @staticmethod
    def calculate_exit_fees(position_size: float) -> Tuple[float, Dict]:
        """Calculate fees for exiting delta neutral position"""
        # Same as entry
        return FeeCalculator.calculate_entry_fees(position_size)

    @staticmethod
    def calculate_round_trip_fees(position_size: float) -> Tuple[float, Dict]:
        """Calculate total round-trip fees"""
        entry_fees, entry_breakdown = FeeCalculator.calculate_entry_fees(position_size)
        exit_fees, exit_breakdown = FeeCalculator.calculate_exit_fees(position_size)

        total = entry_fees + exit_fees

        breakdown = {
            "entry": entry_breakdown,
            "exit": exit_breakdown,
            "total": total
        }

        return total, breakdown

    @staticmethod
    def calculate_funding_profit(funding_rate: float, position_size: float, periods: int = 1) -> float:
        """
        Calculate expected funding profit

        Args:
            funding_rate: Per 8-hour funding rate
            position_size: Position size in USD
            periods: Number of 8-hour periods (1 = 8h, 3 = 24h, 90 = 30 days)
        """
        return funding_rate * position_size * periods

    @staticmethod
    def calculate_payback_period(funding_rate: float, position_size: float) -> float:
        """
        Calculate days to break even on fees

        Returns: days (float)
        """
        round_trip_fees, _ = FeeCalculator.calculate_round_trip_fees(position_size)
        daily_funding = funding_rate * position_size * 3  # 3 periods per day

        if daily_funding <= 0:
            return float('inf')

        return round_trip_fees / daily_funding

# ====================================================================
# DATA PROCESSOR
# ====================================================================

class DataProcessor:
    """Process and buffer WebSocket data"""

    def __init__(self):
        self.funding_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self.trade_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self.cvd_state: Dict[str, float] = {}  # symbol -> current CVD
        self.last_process_time = datetime.now()

    def process_funding_message(self, msg: Dict) -> Optional[FundingData]:
        """Process @markPrice message"""
        try:
            data = FundingData(
                timestamp=datetime.fromtimestamp(msg["E"] / 1000),
                symbol=msg["s"],
                funding_rate=float(msg["r"]),
                mark_price=float(msg["p"]),
                next_funding_time=datetime.fromtimestamp(msg["T"] / 1000)
            )
            self.funding_buffer.append(data)
            return data
        except Exception as e:
            logger.error(f"Error processing funding message: {e}")
            return None

    def process_trade_message(self, msg: Dict) -> Optional[TradeData]:
        """Process @trade message and update CVD"""
        try:
            symbol = msg["s"]
            price = float(msg["p"])
            quantity = float(msg["q"])
            is_buyer_maker = msg["m"]

            # CVD calculation:
            # m = True → seller is aggressor → negative
            # m = False → buyer is aggressor → positive
            signed_volume = -quantity if is_buyer_maker else quantity

            # Update CVD state
            if symbol not in self.cvd_state:
                self.cvd_state[symbol] = 0.0
            self.cvd_state[symbol] += signed_volume

            data = TradeData(
                timestamp=datetime.fromtimestamp(msg["E"] / 1000),
                symbol=symbol,
                price=price,
                quantity=quantity,
                is_buyer_maker=is_buyer_maker,
                signed_volume=signed_volume
            )
            self.trade_buffer.append(data)
            return data
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
            return None

    def create_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert buffers to DataFrames"""
        # Funding DataFrame
        funding_records = []
        for f in self.funding_buffer:
            funding_records.append({
                "timestamp": f.timestamp,
                "symbol": f.symbol,
                "funding_rate": f.funding_rate,
                "mark_price": f.mark_price
            })

        funding_df = pd.DataFrame(funding_records)
        if not funding_df.empty:
            funding_df.set_index("timestamp", inplace=True)

        # Trade DataFrame
        trade_records = []
        for t in self.trade_buffer:
            trade_records.append({
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "price": t.price,
                "quantity": t.quantity,
                "signed_volume": t.signed_volume,
                "cvd": self.cvd_state.get(t.symbol, 0.0)
            })

        trade_df = pd.DataFrame(trade_records)
        if not trade_df.empty:
            trade_df.set_index("timestamp", inplace=True)

        return funding_df, trade_df

# ====================================================================
# FEATURE ENGINE
# ====================================================================

class FeatureEngine:
    """Calculate features from time series data"""

    @staticmethod
    def resample_to_1min(funding_df: pd.DataFrame, trade_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Resample data to 1-minute bars"""
        # Filter by symbol
        funding_sym = funding_df[funding_df["symbol"] == symbol] if not funding_df.empty else pd.DataFrame()
        trade_sym = trade_df[trade_df["symbol"] == symbol] if not trade_df.empty else pd.DataFrame()

        # Resample funding (last value in each minute)
        if not funding_sym.empty:
            funding_1m = funding_sym.resample("1min").last()[["funding_rate", "mark_price"]]
        else:
            funding_1m = pd.DataFrame()

        # Resample trades (aggregations)
        if not trade_sym.empty:
            trade_1m = trade_sym.resample("1min").agg({
                "price": "last",
                "quantity": "sum",
                "signed_volume": "sum",
                "cvd": "last"
            })
        else:
            trade_1m = pd.DataFrame()

        # Merge
        if not funding_1m.empty and not trade_1m.empty:
            merged = funding_1m.join(trade_1m, how="outer")
        elif not funding_1m.empty:
            merged = funding_1m
        elif not trade_1m.empty:
            merged = trade_1m
        else:
            merged = pd.DataFrame()

        # Forward fill for missing values
        if not merged.empty:
            merged.fillna(method="ffill", inplace=True)

        return merged

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features"""
        if df.empty or len(df) < 2:
            return df

        # Funding features
        if "funding_rate" in df.columns:
            # Rolling mean and std (1 hour = 60 minutes)
            df["funding_mean_1h"] = df["funding_rate"].rolling(window=ROLLING_WINDOW_1H, min_periods=1).mean()
            df["funding_std_1h"] = df["funding_rate"].rolling(window=ROLLING_WINDOW_1H, min_periods=1).std()

            # Z-score
            df["funding_z"] = (df["funding_rate"] - df["funding_mean_1h"]) / (df["funding_std_1h"] + 1e-9)

            # Change
            df["funding_change_1h"] = df["funding_rate"] - df["funding_rate"].shift(60)
            df["funding_change_15m"] = df["funding_rate"] - df["funding_rate"].shift(15)

        # CVD features
        if "cvd" in df.columns:
            df["cvd_change_15m"] = df["cvd"] - df["cvd"].shift(ROLLING_WINDOW_15M)
            df["cvd_change_1h"] = df["cvd"] - df["cvd"].shift(ROLLING_WINDOW_1H)

            # CVD momentum
            df["cvd_momentum"] = df["cvd"].diff(5).rolling(window=5).mean()

        # Price features
        if "price" in df.columns:
            df["returns_1m"] = df["price"].pct_change()
            df["returns_15m"] = df["price"].pct_change(15)
            df["volatility_1h"] = df["returns_1m"].rolling(window=60).std()

        return df

    @staticmethod
    def calculate_basis(perp_price: float, spot_price: float) -> float:
        """Calculate basis (perp premium over spot)"""
        if spot_price == 0:
            return 0.0
        return (perp_price / spot_price) - 1.0

# ====================================================================
# SIGNAL GENERATOR
# ====================================================================

class SignalGenerator:
    """Generate trading signals from features"""

    @staticmethod
    def generate_signal(df: pd.DataFrame, symbol: str, spot_price: float) -> Optional[Signal]:
        """
        Generate signal based on current market conditions

        Signal Logic:
        1. HIGH FUNDING + POSITIVE CVD → SHORT (funding arbitrage)
        2. LOW FUNDING + NEGATIVE CVD → LONG (contrarian)
        3. EXTREME FUNDING SPIKE → Mean reversion opportunity
        """
        if df.empty or len(df) < ROLLING_WINDOW_15M:
            return None

        # Get latest values
        latest = df.iloc[-1]

        funding_rate = latest.get("funding_rate", 0)
        funding_z = latest.get("funding_z", 0)
        cvd_change_15m = latest.get("cvd_change_15m", 0)
        mark_price = latest.get("mark_price", latest.get("price", 0))

        # Calculate basis
        basis = FeatureEngine.calculate_basis(mark_price, spot_price)

        # Calculate expected profit and fees
        daily_funding_profit = FeeCalculator.calculate_funding_profit(funding_rate, POSITION_SIZE_USD, periods=3)
        round_trip_fees, _ = FeeCalculator.calculate_round_trip_fees(POSITION_SIZE_USD)
        payback_days = FeeCalculator.calculate_payback_period(funding_rate, POSITION_SIZE_USD)

        # Signal conditions
        signal_type = None
        confidence = 0.0

        # Condition 1: High funding + positive basis → SHORT
        if funding_z > FUNDING_Z_THRESHOLD and funding_rate > 0.0005:  # >0.05% per 8h
            if cvd_change_15m > CVD_CHANGE_THRESHOLD:  # Aggressive longs
                signal_type = "SHORT"
                confidence = min(abs(funding_z) / 5.0, 1.0)  # Scale confidence

        # Condition 2: Negative funding + negative CVD → LONG
        elif funding_z < -FUNDING_Z_THRESHOLD and funding_rate < -0.0005:
            if cvd_change_15m < -CVD_CHANGE_THRESHOLD:  # Aggressive shorts
                signal_type = "LONG"
                confidence = min(abs(funding_z) / 5.0, 1.0)

        # Condition 3: Extreme funding spike (mean reversion)
        elif abs(funding_z) > FUNDING_Z_THRESHOLD * 1.5:
            signal_type = "SHORT" if funding_rate > 0 else "LONG"
            confidence = min(abs(funding_z) / 7.0, 1.0)

        # Only generate signal if profitable
        if signal_type and payback_days < 7:  # Must break even within 1 week
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                funding_rate=funding_rate,
                cvd_change=cvd_change_15m,
                basis=basis,
                expected_profit_daily=daily_funding_profit,
                fees_total=round_trip_fees,
                payback_days=payback_days
            )

        return None

# ====================================================================
# WEBSOCKET MANAGER
# ====================================================================

class WebSocketManager:
    """Manage WebSocket connections and data flow"""

    def __init__(self, symbols: List[str], telegram_notifier: TelegramNotifier):
        self.symbols = symbols
        self.processor = DataProcessor()
        self.telegram = telegram_notifier
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.running = False
        self.event_count = 0
        self.last_summary_time = datetime.now()

    def build_stream_url(self) -> str:
        """Build WebSocket URL for multiple symbols"""
        streams = []
        for symbol in self.symbols:
            sym_lower = symbol.lower()
            streams.append(f"{sym_lower}@markPrice")
            streams.append(f"{sym_lower}@trade")

        stream_str = "/".join(streams)
        return f"wss://fstream.binance.com/stream?streams={stream_str}"

    async def process_message(self, message: Dict):
        """Process incoming WebSocket message"""
        try:
            data = message.get("data")
            if not data:
                return

            event_type = data.get("e")

            if event_type == "markPriceUpdate":
                self.processor.process_funding_message(data)
            elif event_type == "trade":
                self.processor.process_trade_message(data)

            self.event_count += 1

            # Process buffer periodically
            if self.event_count >= BUFFER_SIZE:
                await self.analyze_and_signal()
                self.event_count = 0

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def analyze_and_signal(self):
        """Analyze data and generate signals"""
        try:
            # Create DataFrames
            funding_df, trade_df = self.processor.create_dataframe()

            if funding_df.empty and trade_df.empty:
                return

            # Analyze each symbol
            for symbol in self.symbols:
                # Resample to 1-minute bars
                df_1m = FeatureEngine.resample_to_1min(funding_df, trade_df, symbol)

                if df_1m.empty or len(df_1m) < ROLLING_WINDOW_15M:
                    continue

                # Calculate features
                df_1m = FeatureEngine.calculate_features(df_1m)

                # Get spot price (for basis calculation)
                spot_price = self.get_spot_price(symbol)

                # Generate signal
                signal = SignalGenerator.generate_signal(df_1m, symbol, spot_price)

                if signal:
                    logger.info(f"🚨 SIGNAL: {signal.signal_type} {signal.symbol} | "
                               f"Confidence: {signal.confidence:.1%} | "
                               f"Funding: {signal.funding_rate*100:.4f}% | "
                               f"Payback: {signal.payback_days:.1f} days")

                    # Send Telegram notification
                    self.telegram.send_signal(signal)

                # Print dashboard
                self.print_dashboard(df_1m, symbol)

            # Send periodic summary
            if (datetime.now() - self.last_summary_time).total_seconds() > 3600:  # Every hour
                await self.send_summary()
                self.last_summary_time = datetime.now()

        except Exception as e:
            logger.error(f"Error in analyze_and_signal: {e}")

    def get_spot_price(self, symbol: str) -> float:
        """Get spot price for basis calculation"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except:
            return 0.0

    def print_dashboard(self, df: pd.DataFrame, symbol: str):
        """Print mini dashboard to console"""
        if df.empty or len(df) < 5:
            return

        print("\n" + "="*80)
        print(f"📊 DASHBOARD: {symbol} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Last 5 bars
        tail = df.tail(5)

        print("\nLAST 5 BARS (1-minute):")
        cols_to_show = ["funding_rate", "funding_z", "cvd", "cvd_change_15m", "mark_price"]
        available_cols = [c for c in cols_to_show if c in tail.columns]

        if available_cols:
            print(tail[available_cols].to_string())

        # Current values
        latest = df.iloc[-1]
        print(f"\n📈 CURRENT VALUES:")
        print(f"  Funding Rate: {latest.get('funding_rate', 0)*100:.4f}% per 8h")
        print(f"  Funding Z-Score: {latest.get('funding_z', 0):.2f}")
        print(f"  CVD: {latest.get('cvd', 0):,.0f}")
        print(f"  CVD Change (15m): {latest.get('cvd_change_15m', 0):,.0f}")
        print(f"  Mark Price: ${latest.get('mark_price', latest.get('price', 0)):,.2f}")

        # Profit calculations
        funding_rate = latest.get("funding_rate", 0)
        daily_profit = FeeCalculator.calculate_funding_profit(funding_rate, POSITION_SIZE_USD, periods=3)
        fees, fee_breakdown = FeeCalculator.calculate_round_trip_fees(POSITION_SIZE_USD)
        payback = FeeCalculator.calculate_payback_period(funding_rate, POSITION_SIZE_USD)

        print(f"\n💰 PROFIT ANALYSIS (${POSITION_SIZE_USD:,} position):")
        print(f"  Daily Funding Profit: ${daily_profit:.2f}")
        print(f"  Round-trip Fees: ${fees:.2f}")
        print(f"  Payback Period: {payback:.1f} days")

        if payback < 7:
            print(f"  ✅ PROFITABLE - Break even in {payback:.1f} days")
        else:
            print(f"  ⚠️  MARGINAL - Break even takes {payback:.1f} days")

        print("="*80 + "\n")

    async def send_summary(self):
        """Send hourly summary to Telegram"""
        funding_df, _ = self.processor.create_dataframe()

        if funding_df.empty:
            return

        # Calculate stats
        avg_funding = funding_df["funding_rate"].mean() if "funding_rate" in funding_df.columns else 0
        max_funding = funding_df["funding_rate"].max() if "funding_rate" in funding_df.columns else 0

        # Top opportunities
        top_symbols = funding_df.groupby("symbol")["funding_rate"].last().sort_values(ascending=False).head(3)
        top_str = "\n".join([f"• {sym}: {rate*100:.4f}%" for sym, rate in top_symbols.items()])

        summary = {
            "avg_funding": avg_funding,
            "max_funding": max_funding,
            "top_opportunities": top_str,
            "active_signals": 0  # TODO: track active positions
        }

        self.telegram.send_summary(summary)

    async def connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        url = self.build_stream_url()
        logger.info(f"Connecting to: {url}")

        while self.running:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info("✅ WebSocket connected")

                    # Send connection notification
                    self.telegram.send_message("🟢 <b>Funding CVD System Started</b>\n\nMonitoring funding rates and order flow...")

                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        await self.process_message(data)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("❌ WebSocket connection closed. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def start(self):
        """Start the system"""
        self.running = True
        logger.info("🚀 Starting Funding CVD Analysis System")
        await self.connect_and_listen()

    def stop(self):
        """Stop the system"""
        self.running = False
        logger.info("🛑 Stopping system...")

# ====================================================================
# MAIN
# ====================================================================

async def main():
    """Main entry point"""
    # Initialize components
    telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    ws_manager = WebSocketManager(SYMBOLS, telegram)

    try:
        # Start system
        await ws_manager.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Keyboard interrupt received")
        ws_manager.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        telegram.send_message(f"🔴 <b>System Error</b>\n\n{str(e)}")
    finally:
        telegram.send_message("🔴 <b>System Stopped</b>")

if __name__ == "__main__":
    asyncio.run(main())
