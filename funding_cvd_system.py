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
MAX_SYMBOLS = 5  # Maximum number of symbols to monitor simultaneously
MIN_FUNDING_RATE = 0.0003  # Minimum funding rate to consider (0.03% per 8h)
UPDATE_SYMBOLS_INTERVAL = 3600  # Update symbol list every hour (seconds)
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
# CVD_CHANGE_THRESHOLD removed - using CVD for direction only, not threshold
BASIS_THRESHOLD = 0.001  # 0.1% basis threshold

# Data Collection
ENABLE_SIGNAL_LOGGING = True  # Log all signals to CSV for analysis
SIGNAL_LOG_FILE = "signals_log.csv"  # Signal log file

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
# SYMBOL SCANNER (Dynamic Symbol Selection)
# ====================================================================

class SymbolScanner:
    """Scan and select symbols with highest funding rates"""

    def __init__(self, client: Client):
        self.client = client

    def get_all_usdt_perpetuals(self) -> List[str]:
        """Get all USDT-M perpetual symbols"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = []
            for symbol_info in exchange_info["symbols"]:
                if (
                    symbol_info.get("contractType") == "PERPETUAL"
                    and symbol_info.get("quoteAsset") == "USDT"
                    and symbol_info.get("status") == "TRADING"
                ):
                    symbols.append(symbol_info["symbol"])
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for a symbol"""
        try:
            funding_data = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if funding_data:
                return float(funding_data[0]["fundingRate"])
            return None
        except Exception as e:
            logger.debug(f"Error getting funding for {symbol}: {e}")
            return None

    def scan_top_funding_symbols(
        self,
        max_symbols: int = MAX_SYMBOLS,
        min_funding: float = MIN_FUNDING_RATE
    ) -> List[Tuple[str, float]]:
        """
        Scan all symbols and return top N by funding rate

        Returns:
            List of (symbol, funding_rate) tuples, sorted by funding rate descending
        """
        logger.info("🔍 Scanning all USDT perpetuals for highest funding rates...")

        # Get all symbols
        all_symbols = self.get_all_usdt_perpetuals()
        logger.info(f"Found {len(all_symbols)} USDT perpetual contracts")

        # Get funding rates
        symbol_funding = []
        for symbol in all_symbols:
            funding = self.get_funding_rate(symbol)
            if funding is not None and abs(funding) >= min_funding:
                symbol_funding.append((symbol, funding))

        # Sort by absolute funding rate (we can trade both positive and negative)
        # But prefer positive funding (short positions collect funding)
        symbol_funding.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top N
        top_symbols = symbol_funding[:max_symbols]

        # Log results
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 TOP {len(top_symbols)} FUNDING OPPORTUNITIES:")
        logger.info(f"{'='*80}")
        for symbol, funding in top_symbols:
            direction = "SHORT (collect)" if funding > 0 else "LONG (collect)"
            daily_rate = funding * 3 * 100  # 3 times per day, as percentage
            logger.info(f"  {symbol:12s} | Funding: {funding*100:+.4f}% | "
                       f"Daily: {daily_rate:+.3f}% | {direction}")
        logger.info(f"{'='*80}\n")

        return top_symbols

    def calculate_expected_profit(
        self,
        symbol: str,
        funding_rate: float,
        position_size: float = POSITION_SIZE_USD
    ) -> Dict:
        """Calculate expected profit for a symbol"""
        # Daily funding (3 times per day)
        daily_funding = funding_rate * position_size * 3

        # Fees
        round_trip_fees, _ = FeeCalculator.calculate_round_trip_fees(position_size)

        # Payback period
        if daily_funding <= 0:
            payback_days = float('inf')
        else:
            payback_days = round_trip_fees / daily_funding

        return {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "daily_profit": daily_funding,
            "monthly_profit": daily_funding * 30,
            "annual_apy": (daily_funding * 365 / position_size) * 100,
            "round_trip_fees": round_trip_fees,
            "payback_days": payback_days
        }

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

    def send_symbol_update(self, old_symbols: List[str], new_symbols: List[str]) -> bool:
        """Send notification when monitored symbols change"""
        removed = set(old_symbols) - set(new_symbols)
        added = set(new_symbols) - set(old_symbols)

        if not removed and not added:
            return False

        message = "🔄 <b>Symbol List Updated</b>\n\n"

        if removed:
            message += "<b>❌ Removed (lower funding):</b>\n"
            for sym in removed:
                message += f"  • {sym}\n"
            message += "\n"

        if added:
            message += "<b>✅ Added (higher funding):</b>\n"
            for sym in added:
                message += f"  • {sym}\n"

        message += f"\n<b>Now monitoring:</b> {', '.join(new_symbols)}"

        return self.send_message(message)

# ====================================================================
# SIGNAL LOGGER (CSV Export for Analysis)
# ====================================================================

class SignalLogger:
    """Log all signals to CSV for later analysis"""

    def __init__(self, log_file: str = SIGNAL_LOG_FILE):
        self.log_file = log_file
        self.enabled = ENABLE_SIGNAL_LOGGING

        # Create CSV with headers if it doesn't exist
        if self.enabled:
            self._initialize_csv()

    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        import os
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,symbol,signal_type,confidence,funding_rate,funding_z,"
                       "cvd_change_15m,cvd_direction,basis,mark_price,spot_price,"
                       "expected_daily_profit,fees_total,payback_days,notes\n")
            logger.info(f"📝 Signal log file created: {self.log_file}")

    def log_signal(self, signal: Signal, spot_price: float, cvd_direction: str, notes: str = ""):
        """
        Log a signal to CSV

        Args:
            signal: Signal object
            spot_price: Spot price at signal time
            cvd_direction: "BULLISH" or "BEARISH" or "NEUTRAL"
            notes: Optional notes about the signal
        """
        if not self.enabled:
            return

        try:
            with open(self.log_file, 'a') as f:
                line = (
                    f"{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{signal.symbol},"
                    f"{signal.signal_type},"
                    f"{signal.confidence:.4f},"
                    f"{signal.funding_rate:.6f},"
                    f"{signal.cvd_change:.0f},"  # This is funding_z in Signal object
                    f"{signal.cvd_change:.0f},"  # Actual CVD change
                    f"{cvd_direction},"
                    f"{signal.basis:.6f},"
                    f"{signal.expected_profit_daily / signal.confidence if signal.confidence > 0 else 0:.2f},"  # mark_price placeholder
                    f"{spot_price:.2f},"
                    f"{signal.expected_profit_daily:.2f},"
                    f"{signal.fees_total:.2f},"
                    f"{signal.payback_days:.2f},"
                    f"\"{notes}\"\n"
                )
                f.write(line)
        except Exception as e:
            logger.error(f"Error logging signal: {e}")

    def log_signal_detailed(self, **kwargs):
        """
        Log signal with custom fields

        Usage:
            logger.log_signal_detailed(
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                signal_type="SHORT",
                ...
            )
        """
        if not self.enabled:
            return

        try:
            with open(self.log_file, 'a') as f:
                line = (
                    f"{kwargs.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{kwargs.get('symbol', '')},"
                    f"{kwargs.get('signal_type', '')},"
                    f"{kwargs.get('confidence', 0):.4f},"
                    f"{kwargs.get('funding_rate', 0):.6f},"
                    f"{kwargs.get('funding_z', 0):.2f},"
                    f"{kwargs.get('cvd_change_15m', 0):.0f},"
                    f"{kwargs.get('cvd_direction', '')},"
                    f"{kwargs.get('basis', 0):.6f},"
                    f"{kwargs.get('mark_price', 0):.2f},"
                    f"{kwargs.get('spot_price', 0):.2f},"
                    f"{kwargs.get('expected_daily_profit', 0):.2f},"
                    f"{kwargs.get('fees_total', 0):.2f},"
                    f"{kwargs.get('payback_days', 0):.2f},"
                    f"\"{kwargs.get('notes', '')}\"\n"
                )
                f.write(line)
        except Exception as e:
            logger.error(f"Error logging detailed signal: {e}")

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
    def generate_signal(df: pd.DataFrame, symbol: str, spot_price: float) -> Optional[Tuple[Signal, str]]:
        """
        Generate signal based on current market conditions

        SIMPLIFIED Signal Logic (Data-Driven Approach):
        1. HIGH FUNDING (z > 2.0) + Positive CVD → SHORT (contrarian to bullish pressure)
        2. LOW FUNDING (z < -2.0) + Negative CVD → LONG (contrarian to bearish pressure)
        3. EXTREME FUNDING SPIKE (|z| > 3.0) → Mean reversion (ignore CVD)

        CVD is used for DIRECTION only (no threshold):
        - CVD > 0 = Bullish pressure (buyers aggressive)
        - CVD < 0 = Bearish pressure (sellers aggressive)

        Returns:
            Tuple[Signal, cvd_direction] or None
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

        # CVD Direction (no threshold, just direction)
        cvd_direction = "BULLISH" if cvd_change_15m > 0 else "BEARISH" if cvd_change_15m < 0 else "NEUTRAL"

        # Signal conditions
        signal_type = None
        confidence = 0.0
        signal_reason = ""

        # Condition 1: High funding + bullish CVD → SHORT (contrarian)
        if funding_z > FUNDING_Z_THRESHOLD and funding_rate > 0.0005:  # >0.05% per 8h
            if cvd_change_15m > 0:  # Bullish pressure (no threshold)
                signal_type = "SHORT"
                confidence = min(abs(funding_z) / 5.0, 1.0)
                signal_reason = "HIGH_FUNDING_BULLISH_CVD"

        # Condition 2: Negative funding + bearish CVD → LONG (contrarian)
        elif funding_z < -FUNDING_Z_THRESHOLD and funding_rate < -0.0005:
            if cvd_change_15m < 0:  # Bearish pressure (no threshold)
                signal_type = "LONG"
                confidence = min(abs(funding_z) / 5.0, 1.0)
                signal_reason = "LOW_FUNDING_BEARISH_CVD"

        # Condition 3: Extreme funding spike (mean reversion, ignore CVD)
        elif abs(funding_z) > FUNDING_Z_THRESHOLD * 1.5:
            signal_type = "SHORT" if funding_rate > 0 else "LONG"
            confidence = min(abs(funding_z) / 7.0, 1.0)
            signal_reason = "EXTREME_FUNDING_SPIKE"

        # Only generate signal if profitable
        if signal_type and payback_days < 7:  # Must break even within 1 week
            signal = Signal(
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
            return (signal, cvd_direction, signal_reason)

        return None

# ====================================================================
# WEBSOCKET MANAGER
# ====================================================================

class WebSocketManager:
    """Manage WebSocket connections and data flow"""

    def __init__(self, telegram_notifier: TelegramNotifier):
        self.symbols = []  # Will be populated dynamically
        self.processor = DataProcessor()
        self.telegram = telegram_notifier
        self.signal_logger = SignalLogger()  # CSV logger for analysis
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.scanner = SymbolScanner(self.client)
        self.running = False
        self.event_count = 0
        self.last_summary_time = datetime.now()
        self.last_symbol_update = datetime.now()
        self.websocket = None  # Current WebSocket connection

    def update_symbols(self) -> bool:
        """
        Update symbol list based on current funding rates

        Returns:
            True if symbols changed, False otherwise
        """
        # Scan for top funding symbols
        top_funding = self.scanner.scan_top_funding_symbols(
            max_symbols=MAX_SYMBOLS,
            min_funding=MIN_FUNDING_RATE
        )

        # Extract symbol names
        new_symbols = [symbol for symbol, _ in top_funding]

        # Check if changed
        if set(new_symbols) == set(self.symbols):
            logger.info("✓ Symbol list unchanged")
            return False

        # Send Telegram notification
        if self.symbols:  # Not first time
            self.telegram.send_symbol_update(self.symbols, new_symbols)

        # Send funding summary to Telegram
        summary_msg = "📊 <b>Top Funding Opportunities</b>\n\n"
        for symbol, funding in top_funding:
            profit_info = self.scanner.calculate_expected_profit(symbol, funding)
            direction = "🔴 SHORT" if funding > 0 else "🟢 LONG"
            summary_msg += f"{direction} <b>{symbol}</b>\n"
            summary_msg += f"  • Funding: {funding*100:+.4f}% per 8h\n"
            summary_msg += f"  • Daily: ${profit_info['daily_profit']:.2f}\n"
            summary_msg += f"  • Payback: {profit_info['payback_days']:.1f} days\n\n"

        self.telegram.send_message(summary_msg)

        # Update symbols
        old_symbols = self.symbols
        self.symbols = new_symbols

        logger.info(f"✓ Updated symbols: {', '.join(new_symbols)}")

        return True

    def build_stream_url(self) -> str:
        """Build WebSocket URL for multiple symbols"""
        if not self.symbols:
            raise ValueError("No symbols to monitor")

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

                # Generate signal (returns tuple or None)
                signal_result = SignalGenerator.generate_signal(df_1m, symbol, spot_price)

                if signal_result:
                    signal, cvd_direction, signal_reason = signal_result

                    logger.info(f"🚨 SIGNAL: {signal.signal_type} {signal.symbol} | "
                               f"Confidence: {signal.confidence:.1%} | "
                               f"Funding: {signal.funding_rate*100:.4f}% | "
                               f"CVD: {cvd_direction} | "
                               f"Reason: {signal_reason} | "
                               f"Payback: {signal.payback_days:.1f} days")

                    # Send Telegram notification
                    self.telegram.send_signal(signal)

                    # Log to CSV for analysis
                    self.signal_logger.log_signal_detailed(
                        timestamp=signal.timestamp,
                        symbol=signal.symbol,
                        signal_type=signal.signal_type,
                        confidence=signal.confidence,
                        funding_rate=signal.funding_rate,
                        funding_z=df_1m.iloc[-1].get("funding_z", 0),
                        cvd_change_15m=signal.cvd_change,
                        cvd_direction=cvd_direction,
                        basis=signal.basis,
                        mark_price=df_1m.iloc[-1].get("mark_price", 0),
                        spot_price=spot_price,
                        expected_daily_profit=signal.expected_profit_daily,
                        fees_total=signal.fees_total,
                        payback_days=signal.payback_days,
                        notes=signal_reason
                    )

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
        # Initial symbol update
        self.update_symbols()

        while self.running:
            try:
                # Build URL with current symbols
                url = self.build_stream_url()
                logger.info(f"Connecting to: {url}")

                async with websockets.connect(url) as websocket:
                    self.websocket = websocket
                    logger.info("✅ WebSocket connected")

                    # Send connection notification
                    if not self.last_symbol_update or (datetime.now() - self.last_symbol_update).total_seconds() < 10:
                        self.telegram.send_message("🟢 <b>Funding CVD System Started</b>\n\nMonitoring funding rates and order flow...")

                    while self.running:
                        # Check if symbols need updating
                        if (datetime.now() - self.last_symbol_update).total_seconds() > UPDATE_SYMBOLS_INTERVAL:
                            logger.info("⏰ Time to update symbols...")
                            symbols_changed = self.update_symbols()
                            self.last_symbol_update = datetime.now()

                            if symbols_changed:
                                logger.info("🔄 Symbols changed, reconnecting WebSocket...")
                                break  # Break to reconnect with new symbols

                        # Receive message with timeout
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            data = json.loads(message)
                            await self.process_message(data)
                        except asyncio.TimeoutError:
                            # Timeout is normal, just check for symbol updates
                            continue

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
    ws_manager = WebSocketManager(telegram)

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
