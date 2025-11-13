"""
Multi-Regime Market Analysis System
====================================

4 Market Regimes:
1. Spot-Led Trend (güvenilir trend - en temiz para)
2. Perp-Led Euphoria (şişik piyasa - fade opportunity)
3. Spot Accumulation (gizli hazırlık - erken giriş)
4. Funding Carry (carry trade - delta neutral)

Data Sources:
- Spot CVD (Binance Spot trades)
- Perp CVD (Binance Futures trades)
- Open Interest (Binance Futures)
- Funding Rate (real-time)
- Basis (perp vs spot premium)

Author: Claude
Date: 2025-01-13
"""

import asyncio
import websockets
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import pandas as pd
import numpy as np
from binance.client import Client

# ====================================================================
# CONFIGURATION
# ====================================================================

# Binance API
BINANCE_API_KEY = "YOUR_API_KEY"
BINANCE_API_SECRET = "YOUR_SECRET"

# Telegram
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# Symbol Selection
MAX_SYMBOLS = 5  # Maximum symbols to monitor
MIN_FUNDING_RATE = 0.0003  # Minimum 0.03% per 8h
MIN_VOLUME_24H = 50000000  # Minimum $50M daily volume

# Analysis Parameters
SPOT_CVD_WINDOW = 15  # 15 minutes for spot CVD slope
PERP_CVD_WINDOW = 15  # 15 minutes for perp CVD slope
OI_WINDOW = 60  # 60 minutes for OI change detection
FUNDING_Z_WINDOW = 60  # 60 minutes for funding z-score

# Regime Thresholds
SPOT_CVD_STRONG_THRESHOLD = 0.3  # 30% net directional flow
PERP_CVD_STRONG_THRESHOLD = 0.5  # 50% aggressive flow
OI_CHANGE_THRESHOLD = 0.1  # 10% OI change
FUNDING_HIGH_THRESHOLD = 0.0005  # 0.05% per 8h
BASIS_HIGH_THRESHOLD = 0.002  # 0.2% basis

# Data Collection
BUFFER_SIZE = 1000  # Number of events to buffer
UPDATE_INTERVAL = 3600  # Symbol update interval (1 hour)
REGIME_LOG_FILE = "regime_stats.csv"
SIGNAL_LOG_FILE = "regime_signals.csv"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================================================================
# ENUMS & DATA STRUCTURES
# ====================================================================

class MarketRegime(Enum):
    """4 Market Regimes"""
    SPOT_LED_TREND = "SPOT_LED_TREND"  # 🟢 Reliable trend
    PERP_LED_EUPHORIA = "PERP_LED_EUPHORIA"  # 🟡 Overheated
    SPOT_ACCUMULATION = "SPOT_ACCUMULATION"  # 🟢 Early accumulation
    FUNDING_CARRY = "FUNDING_CARRY"  # 🔵 Carry trade
    UNKNOWN = "UNKNOWN"  # Not enough data

@dataclass
class MarketData:
    """Market data snapshot"""
    timestamp: datetime
    symbol: str
    spot_price: float
    perp_price: float
    spot_cvd: float
    perp_cvd: float
    open_interest: float
    funding_rate: float
    basis: float

@dataclass
class RegimeState:
    """Current regime state"""
    regime: MarketRegime
    timestamp: datetime
    symbol: str
    spot_cvd_slope: float
    perp_cvd_slope: float
    oi_change_pct: float
    funding_z: float
    basis: float
    confidence: float  # 0-1

@dataclass
class RegimeSignal:
    """Trading signal based on regime"""
    timestamp: datetime
    symbol: str
    regime: MarketRegime
    signal_type: str  # "LONG", "SHORT", "NEUTRAL", "CARRY"
    confidence: float
    entry_reason: str
    expected_holding_days: float
    notes: str

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
            logger.error(f"Telegram error: {e}")
            return False

    def send_regime_change(self, old_regime: MarketRegime, new_regime: RegimeState) -> bool:
        """Notify regime change"""
        emoji_map = {
            MarketRegime.SPOT_LED_TREND: "🟢",
            MarketRegime.PERP_LED_EUPHORIA: "🟡",
            MarketRegime.SPOT_ACCUMULATION: "🟢",
            MarketRegime.FUNDING_CARRY: "🔵",
            MarketRegime.UNKNOWN: "⚪"
        }

        old_emoji = emoji_map.get(old_regime, "⚪")
        new_emoji = emoji_map.get(new_regime.regime, "⚪")

        message = f"""
{old_emoji}→{new_emoji} <b>REGIME CHANGE: {new_regime.symbol}</b>

<b>Old:</b> {old_regime.value if old_regime != MarketRegime.UNKNOWN else 'UNKNOWN'}
<b>New:</b> {new_regime.regime.value}
<b>Confidence:</b> {new_regime.confidence:.1%}

<b>Metrics:</b>
• Spot CVD Slope: {new_regime.spot_cvd_slope:+.2f}
• Perp CVD Slope: {new_regime.perp_cvd_slope:+.2f}
• OI Change: {new_regime.oi_change_pct:+.1%}
• Funding Z: {new_regime.funding_z:+.2f}
• Basis: {new_regime.basis*100:+.3f}%

⏰ {new_regime.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)

    def send_signal(self, signal: RegimeSignal) -> bool:
        """Send trading signal"""
        emoji_map = {
            MarketRegime.SPOT_LED_TREND: "🟢",
            MarketRegime.PERP_LED_EUPHORIA: "🟡",
            MarketRegime.SPOT_ACCUMULATION: "🟢",
            MarketRegime.FUNDING_CARRY: "🔵"
        }

        emoji = emoji_map.get(signal.regime, "⚪")

        message = f"""
{emoji} <b>{signal.signal_type} SIGNAL: {signal.symbol}</b>

<b>Regime:</b> {signal.regime.value}
<b>Confidence:</b> {signal.confidence:.1%}
<b>Expected Hold:</b> {signal.expected_holding_days:.1f} days

<b>Entry Reason:</b>
{signal.entry_reason}

<b>Notes:</b>
{signal.notes}

⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)

    def send_hourly_summary(self, regime_stats: Dict) -> bool:
        """Send hourly regime statistics"""
        message = f"""
📊 <b>HOURLY REGIME SUMMARY</b>

<b>Time in Each Regime (last hour):</b>
"""
        for regime, pct in regime_stats.get('time_distribution', {}).items():
            message += f"  • {regime}: {pct:.1f}%\n"

        message += f"""
<b>Signals Generated:</b> {regime_stats.get('signals_count', 0)}

<b>Top Symbols by Activity:</b>
"""
        for symbol, count in regime_stats.get('top_symbols', []):
            message += f"  • {symbol}: {count} regime changes\n"

        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

# ====================================================================
# STATISTICS LOGGER
# ====================================================================

class StatisticsLogger:
    """Log regime statistics for analysis"""

    def __init__(self, regime_log: str = REGIME_LOG_FILE, signal_log: str = SIGNAL_LOG_FILE):
        self.regime_log = regime_log
        self.signal_log = signal_log
        self._initialize_logs()

    def _initialize_logs(self):
        """Create log files with headers"""
        import os

        # Regime log
        if not os.path.exists(self.regime_log):
            with open(self.regime_log, 'w') as f:
                f.write("timestamp,symbol,regime,confidence,spot_cvd_slope,perp_cvd_slope,"
                       "oi_change_pct,funding_z,basis,duration_seconds\n")
            logger.info(f"📝 Regime log created: {self.regime_log}")

        # Signal log
        if not os.path.exists(self.signal_log):
            with open(self.signal_log, 'w') as f:
                f.write("timestamp,symbol,regime,signal_type,confidence,entry_reason,"
                       "expected_holding_days,notes\n")
            logger.info(f"📝 Signal log created: {self.signal_log}")

    def log_regime(self, state: RegimeState, duration: float = 0):
        """Log regime state"""
        try:
            with open(self.regime_log, 'a') as f:
                f.write(f"{state.timestamp.strftime('%Y-%m-%d %H:%M:%S')},"
                       f"{state.symbol},"
                       f"{state.regime.value},"
                       f"{state.confidence:.4f},"
                       f"{state.spot_cvd_slope:.4f},"
                       f"{state.perp_cvd_slope:.4f},"
                       f"{state.oi_change_pct:.4f},"
                       f"{state.funding_z:.4f},"
                       f"{state.basis:.6f},"
                       f"{duration:.1f}\n")
        except Exception as e:
            logger.error(f"Error logging regime: {e}")

    def log_signal(self, signal: RegimeSignal):
        """Log trading signal"""
        try:
            with open(self.signal_log, 'a') as f:
                f.write(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')},"
                       f"{signal.symbol},"
                       f"{signal.regime.value},"
                       f"{signal.signal_type},"
                       f"{signal.confidence:.4f},"
                       f'"{signal.entry_reason}",'
                       f"{signal.expected_holding_days:.2f},"
                       f'"{signal.notes}"\n')
        except Exception as e:
            logger.error(f"Error logging signal: {e}")

    def get_regime_statistics(self, hours: int = 24) -> Dict:
        """Get regime performance statistics"""
        try:
            df = pd.read_csv(self.regime_log)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter by time
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df['timestamp'] >= cutoff]

            if df.empty:
                return {}

            # Calculate statistics
            stats = {
                'total_samples': len(df),
                'regime_distribution': df['regime'].value_counts().to_dict(),
                'avg_confidence_by_regime': df.groupby('regime')['confidence'].mean().to_dict(),
                'total_time_by_regime': df.groupby('regime')['duration_seconds'].sum().to_dict(),
                'symbol_counts': df['symbol'].value_counts().head(10).to_dict()
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

# ====================================================================
# SYMBOL SCANNER
# ====================================================================

class SymbolScanner:
    """Scan and select top symbols"""

    def __init__(self, client: Client):
        self.client = client

    def scan_top_symbols(self, max_symbols: int = MAX_SYMBOLS) -> List[str]:
        """
        Scan all USDT perpetuals and return top N by:
        1. High funding rate
        2. High volume
        3. Adequate liquidity
        """
        try:
            logger.info("🔍 Scanning symbols...")

            # Get all USDT perpetuals
            exchange_info = self.client.futures_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']
                      if s.get('contractType') == 'PERPETUAL'
                      and s.get('quoteAsset') == 'USDT'
                      and s.get('status') == 'TRADING']

            # Get 24h tickers for volume filtering
            tickers = self.client.futures_ticker()
            ticker_map = {t['symbol']: float(t['quoteVolume']) for t in tickers}

            # Filter by volume
            symbols = [s for s in symbols if ticker_map.get(s, 0) >= MIN_VOLUME_24H]

            # Get funding rates
            symbol_funding = []
            for symbol in symbols[:50]:  # Limit to top 50 by volume for speed
                try:
                    funding_data = self.client.futures_funding_rate(symbol=symbol, limit=1)
                    if funding_data:
                        funding = float(funding_data[0]['fundingRate'])
                        if abs(funding) >= MIN_FUNDING_RATE:
                            symbol_funding.append((symbol, funding, ticker_map.get(symbol, 0)))
                except:
                    continue

            # Sort by absolute funding rate
            symbol_funding.sort(key=lambda x: abs(x[1]), reverse=True)

            # Take top N
            top_symbols = [s[0] for s in symbol_funding[:max_symbols]]

            logger.info(f"✅ Selected {len(top_symbols)} symbols:")
            for symbol, funding, volume in symbol_funding[:max_symbols]:
                logger.info(f"  • {symbol}: Funding={funding*100:.4f}%, Volume=${volume/1e6:.1f}M")

            return top_symbols

        except Exception as e:
            logger.error(f"Error scanning symbols: {e}")
            return []

# ====================================================================
# DATA PROCESSOR
# ====================================================================

class DataProcessor:
    """Process real-time market data"""

    def __init__(self):
        # Buffers for each symbol
        self.spot_trades: Dict[str, deque] = {}  # symbol -> deque of trades
        self.perp_trades: Dict[str, deque] = {}
        self.funding_data: Dict[str, deque] = {}
        self.oi_data: Dict[str, deque] = {}

        # Current state
        self.spot_cvd: Dict[str, float] = {}  # symbol -> CVD
        self.perp_cvd: Dict[str, float] = {}
        self.last_oi: Dict[str, float] = {}

    def init_symbol(self, symbol: str):
        """Initialize buffers for a symbol"""
        if symbol not in self.spot_trades:
            self.spot_trades[symbol] = deque(maxlen=BUFFER_SIZE)
            self.perp_trades[symbol] = deque(maxlen=BUFFER_SIZE)
            self.funding_data[symbol] = deque(maxlen=BUFFER_SIZE)
            self.oi_data[symbol] = deque(maxlen=BUFFER_SIZE)
            self.spot_cvd[symbol] = 0.0
            self.perp_cvd[symbol] = 0.0
            self.last_oi[symbol] = 0.0

    def process_spot_trade(self, symbol: str, msg: Dict):
        """Process spot trade message"""
        try:
            timestamp = datetime.fromtimestamp(msg['T'] / 1000)
            price = float(msg['p'])
            qty = float(msg['q'])
            is_buyer_maker = msg['m']

            # CVD calculation
            signed_volume = -qty if is_buyer_maker else qty
            self.spot_cvd[symbol] += signed_volume

            # Store trade
            self.spot_trades[symbol].append({
                'timestamp': timestamp,
                'price': price,
                'qty': qty,
                'signed_volume': signed_volume,
                'cvd': self.spot_cvd[symbol]
            })

        except Exception as e:
            logger.error(f"Error processing spot trade: {e}")

    def process_perp_trade(self, symbol: str, msg: Dict):
        """Process perp trade message"""
        try:
            timestamp = datetime.fromtimestamp(msg['T'] / 1000)
            price = float(msg['p'])
            qty = float(msg['q'])
            is_buyer_maker = msg['m']

            # CVD calculation
            signed_volume = -qty if is_buyer_maker else qty
            self.perp_cvd[symbol] += signed_volume

            # Store trade
            self.perp_trades[symbol].append({
                'timestamp': timestamp,
                'price': price,
                'qty': qty,
                'signed_volume': signed_volume,
                'cvd': self.perp_cvd[symbol]
            })

        except Exception as e:
            logger.error(f"Error processing perp trade: {e}")

    def process_funding(self, symbol: str, msg: Dict):
        """Process funding rate update"""
        try:
            timestamp = datetime.fromtimestamp(msg['E'] / 1000)
            funding_rate = float(msg['r'])
            mark_price = float(msg['p'])

            self.funding_data[symbol].append({
                'timestamp': timestamp,
                'funding_rate': funding_rate,
                'mark_price': mark_price
            })

        except Exception as e:
            logger.error(f"Error processing funding: {e}")

    def update_oi(self, symbol: str, oi: float):
        """Update open interest"""
        self.oi_data[symbol].append({
            'timestamp': datetime.now(),
            'oi': oi
        })
        self.last_oi[symbol] = oi

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market snapshot"""
        try:
            if not self.spot_trades.get(symbol) or not self.perp_trades.get(symbol):
                return None

            # Latest prices
            spot_price = self.spot_trades[symbol][-1]['price'] if self.spot_trades[symbol] else 0
            perp_price = self.perp_trades[symbol][-1]['price'] if self.perp_trades[symbol] else 0

            # CVD
            spot_cvd = self.spot_cvd.get(symbol, 0)
            perp_cvd = self.perp_cvd.get(symbol, 0)

            # Funding
            funding_rate = self.funding_data[symbol][-1]['funding_rate'] if self.funding_data.get(symbol) else 0

            # OI
            oi = self.last_oi.get(symbol, 0)

            # Basis
            basis = (perp_price / spot_price - 1) if spot_price > 0 else 0

            return MarketData(
                timestamp=datetime.now(),
                symbol=symbol,
                spot_price=spot_price,
                perp_price=perp_price,
                spot_cvd=spot_cvd,
                perp_cvd=perp_cvd,
                open_interest=oi,
                funding_rate=funding_rate,
                basis=basis
            )

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

# ====================================================================
# REGIME DETECTOR
# ====================================================================

class RegimeDetector:
    """Detect market regime based on metrics"""

    @staticmethod
    def calculate_cvd_slope(trades: deque, window_minutes: int = 15) -> float:
        """
        Calculate CVD slope (momentum) over time window

        Returns: normalized slope (-1 to 1 range)
        """
        if not trades or len(trades) < 10:
            return 0.0

        try:
            # Convert to DataFrame
            df = pd.DataFrame(list(trades))
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter by time window
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            df = df[df['timestamp'] >= cutoff]

            if len(df) < 2:
                return 0.0

            # Calculate total volume
            total_volume = df['qty'].sum()
            if total_volume == 0:
                return 0.0

            # Net directional flow
            net_flow = df['signed_volume'].sum()

            # Normalized slope (-1 to 1)
            slope = net_flow / total_volume

            return np.clip(slope, -1, 1)

        except Exception as e:
            logger.error(f"Error calculating CVD slope: {e}")
            return 0.0

    @staticmethod
    def calculate_oi_change(oi_data: deque, window_minutes: int = 60) -> float:
        """Calculate OI percentage change"""
        if not oi_data or len(oi_data) < 2:
            return 0.0

        try:
            df = pd.DataFrame(list(oi_data))
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            df = df[df['timestamp'] >= cutoff]

            if len(df) < 2:
                return 0.0

            oi_start = df.iloc[0]['oi']
            oi_end = df.iloc[-1]['oi']

            if oi_start == 0:
                return 0.0

            return (oi_end - oi_start) / oi_start

        except Exception as e:
            logger.error(f"Error calculating OI change: {e}")
            return 0.0

    @staticmethod
    def calculate_funding_z(funding_data: deque, window_minutes: int = 60) -> float:
        """Calculate funding rate z-score"""
        if not funding_data or len(funding_data) < 10:
            return 0.0

        try:
            df = pd.DataFrame(list(funding_data))
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            df = df[df['timestamp'] >= cutoff]

            if len(df) < 2:
                return 0.0

            funding_mean = df['funding_rate'].mean()
            funding_std = df['funding_rate'].std()

            if funding_std == 0:
                return 0.0

            current_funding = df.iloc[-1]['funding_rate']
            z_score = (current_funding - funding_mean) / funding_std

            return z_score

        except Exception as e:
            logger.error(f"Error calculating funding z: {e}")
            return 0.0

    def detect_regime(self, processor: DataProcessor, symbol: str) -> Optional[RegimeState]:
        """
        Detect current market regime

        4 Regimes:
        1. Spot-Led Trend: Strong spot CVD, perp following, OI rising, funding normal
        2. Perp-Led Euphoria: Weak spot CVD, strong perp CVD, OI exploding, funding/basis extreme
        3. Spot Accumulation: Steady spot CVD, weak perp CVD, OI slowly rising, funding neutral
        4. Funding Carry: High funding, basis wide, CVD divergence
        """
        try:
            # Calculate metrics
            spot_cvd_slope = self.calculate_cvd_slope(
                processor.spot_trades.get(symbol, deque()),
                SPOT_CVD_WINDOW
            )

            perp_cvd_slope = self.calculate_cvd_slope(
                processor.perp_trades.get(symbol, deque()),
                PERP_CVD_WINDOW
            )

            oi_change = self.calculate_oi_change(
                processor.oi_data.get(symbol, deque()),
                OI_WINDOW
            )

            funding_z = self.calculate_funding_z(
                processor.funding_data.get(symbol, deque()),
                FUNDING_Z_WINDOW
            )

            # Get current funding and basis
            market_data = processor.get_market_data(symbol)
            if not market_data:
                return None

            funding_rate = market_data.funding_rate
            basis = market_data.basis

            # Regime detection logic
            regime, confidence = self._classify_regime(
                spot_cvd_slope, perp_cvd_slope, oi_change,
                funding_z, funding_rate, basis
            )

            return RegimeState(
                regime=regime,
                timestamp=datetime.now(),
                symbol=symbol,
                spot_cvd_slope=spot_cvd_slope,
                perp_cvd_slope=perp_cvd_slope,
                oi_change_pct=oi_change,
                funding_z=funding_z,
                basis=basis,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return None

    def _classify_regime(
        self,
        spot_slope: float,
        perp_slope: float,
        oi_change: float,
        funding_z: float,
        funding_rate: float,
        basis: float
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on metrics

        Returns: (regime, confidence)
        """

        # Initialize scores for each regime
        scores = {
            MarketRegime.SPOT_LED_TREND: 0.0,
            MarketRegime.PERP_LED_EUPHORIA: 0.0,
            MarketRegime.SPOT_ACCUMULATION: 0.0,
            MarketRegime.FUNDING_CARRY: 0.0
        }

        # --- Regime 1: Spot-Led Trend ---
        # Strong spot CVD, perp following, OI rising, funding/basis normal
        if abs(spot_slope) > SPOT_CVD_STRONG_THRESHOLD:
            scores[MarketRegime.SPOT_LED_TREND] += 3.0

        if (spot_slope > 0 and perp_slope > 0) or (spot_slope < 0 and perp_slope < 0):
            scores[MarketRegime.SPOT_LED_TREND] += 2.0  # Same direction

        if oi_change > 0.05:  # OI rising
            scores[MarketRegime.SPOT_LED_TREND] += 1.0

        if abs(funding_z) < 2.0 and abs(basis) < BASIS_HIGH_THRESHOLD:
            scores[MarketRegime.SPOT_LED_TREND] += 1.0  # Normal funding/basis

        # --- Regime 2: Perp-Led Euphoria ---
        # Weak spot, strong perp, OI exploding, funding/basis extreme
        if abs(perp_slope) > PERP_CVD_STRONG_THRESHOLD:
            scores[MarketRegime.PERP_LED_EUPHORIA] += 3.0

        if abs(spot_slope) < SPOT_CVD_STRONG_THRESHOLD * 0.5:
            scores[MarketRegime.PERP_LED_EUPHORIA] += 2.0  # Weak spot

        if oi_change > OI_CHANGE_THRESHOLD:
            scores[MarketRegime.PERP_LED_EUPHORIA] += 2.0

        if abs(funding_z) > 2.0 or abs(basis) > BASIS_HIGH_THRESHOLD:
            scores[MarketRegime.PERP_LED_EUPHORIA] += 2.0  # Extreme funding/basis

        # --- Regime 3: Spot Accumulation ---
        # Steady spot, weak perp, OI slowly rising, funding neutral
        if 0.1 < abs(spot_slope) < SPOT_CVD_STRONG_THRESHOLD:
            scores[MarketRegime.SPOT_ACCUMULATION] += 3.0  # Moderate spot

        if abs(perp_slope) < PERP_CVD_STRONG_THRESHOLD * 0.5:
            scores[MarketRegime.SPOT_ACCUMULATION] += 2.0  # Weak perp

        if 0 < oi_change < OI_CHANGE_THRESHOLD:
            scores[MarketRegime.SPOT_ACCUMULATION] += 2.0  # Slowly rising

        if abs(funding_z) < 1.5 and abs(funding_rate) < FUNDING_HIGH_THRESHOLD:
            scores[MarketRegime.SPOT_ACCUMULATION] += 1.0

        # --- Regime 4: Funding Carry ---
        # High funding, basis wide, CVD divergence
        if abs(funding_rate) > FUNDING_HIGH_THRESHOLD * 2:
            scores[MarketRegime.FUNDING_CARRY] += 3.0

        if abs(basis) > BASIS_HIGH_THRESHOLD:
            scores[MarketRegime.FUNDING_CARRY] += 2.0

        if abs(funding_z) > 1.5:
            scores[MarketRegime.FUNDING_CARRY] += 1.0

        # CVD divergence (spot up, perp down or vice versa)
        if (spot_slope > 0 and perp_slope < 0) or (spot_slope < 0 and perp_slope > 0):
            scores[MarketRegime.FUNDING_CARRY] += 2.0

        # Select regime with highest score
        best_regime = max(scores, key=scores.get)
        max_score = scores[best_regime]

        # Calculate confidence (normalize to 0-1)
        confidence = min(max_score / 7.0, 1.0)  # Max possible score ~7-10

        # If confidence too low, return UNKNOWN
        if confidence < 0.3:
            return MarketRegime.UNKNOWN, confidence

        return best_regime, confidence

# ====================================================================
# SIGNAL GENERATOR
# ====================================================================

class SignalGenerator:
    """Generate trading signals based on regime"""

    @staticmethod
    def generate_signal(regime_state: RegimeState) -> Optional[RegimeSignal]:
        """
        Generate trading signal based on regime

        Strategy:
        - Regime 1 (Spot-Led Trend): Follow trend direction
        - Regime 2 (Perp-Led Euphoria): Fade/avoid
        - Regime 3 (Spot Accumulation): Early entry
        - Regime 4 (Funding Carry): Delta-neutral carry
        """

        if regime_state.regime == MarketRegime.SPOT_LED_TREND:
            return SignalGenerator._signal_trend(regime_state)

        elif regime_state.regime == MarketRegime.PERP_LED_EUPHORIA:
            return SignalGenerator._signal_fade(regime_state)

        elif regime_state.regime == MarketRegime.SPOT_ACCUMULATION:
            return SignalGenerator._signal_accumulation(regime_state)

        elif regime_state.regime == MarketRegime.FUNDING_CARRY:
            return SignalGenerator._signal_carry(regime_state)

        return None

    @staticmethod
    def _signal_trend(state: RegimeState) -> Optional[RegimeSignal]:
        """Strategy A: Trend following (safest)"""

        # Minimum confidence
        if state.confidence < 0.5:
            return None

        # Determine direction
        if state.spot_cvd_slope > SPOT_CVD_STRONG_THRESHOLD:
            signal_type = "LONG"
            entry_reason = f"Strong spot accumulation ({state.spot_cvd_slope:+.2f}), perp following ({state.perp_cvd_slope:+.2f})"
        elif state.spot_cvd_slope < -SPOT_CVD_STRONG_THRESHOLD:
            signal_type = "SHORT"
            entry_reason = f"Strong spot distribution ({state.spot_cvd_slope:+.2f}), perp following ({state.perp_cvd_slope:+.2f})"
        else:
            return None

        return RegimeSignal(
            timestamp=state.timestamp,
            symbol=state.symbol,
            regime=state.regime,
            signal_type=signal_type,
            confidence=state.confidence,
            entry_reason=entry_reason,
            expected_holding_days=3.0,
            notes="Reliable trend - safest trade. Use 3-5x leverage."
        )

    @staticmethod
    def _signal_fade(state: RegimeState) -> Optional[RegimeSignal]:
        """Strategy B: Fade euphoria (sniper, low frequency)"""

        # Very strict criteria
        if state.confidence < 0.7:
            return None

        # Only extreme cases
        if abs(state.funding_z) < 2.5:
            return None

        # Fade perp direction
        if state.perp_cvd_slope > PERP_CVD_STRONG_THRESHOLD:
            signal_type = "SHORT"
            entry_reason = f"Perp euphoria ({state.perp_cvd_slope:+.2f}), weak spot ({state.spot_cvd_slope:+.2f}), funding spike ({state.funding_z:+.2f})"
        elif state.perp_cvd_slope < -PERP_CVD_STRONG_THRESHOLD:
            signal_type = "LONG"
            entry_reason = f"Perp panic ({state.perp_cvd_slope:+.2f}), weak spot ({state.spot_cvd_slope:+.2f}), funding spike ({state.funding_z:+.2f})"
        else:
            return None

        return RegimeSignal(
            timestamp=state.timestamp,
            symbol=state.symbol,
            regime=state.regime,
            signal_type=signal_type,
            confidence=state.confidence,
            entry_reason=entry_reason,
            expected_holding_days=1.0,
            notes="⚠️ HIGH RISK fade trade. Small position, tight stop."
        )

    @staticmethod
    def _signal_accumulation(state: RegimeState) -> Optional[RegimeSignal]:
        """Strategy A variant: Early entry"""

        if state.confidence < 0.4:
            return None

        # Determine direction
        if state.spot_cvd_slope > 0.1:
            signal_type = "LONG"
            entry_reason = f"Early accumulation ({state.spot_cvd_slope:+.2f}), perp not yet participating"
        elif state.spot_cvd_slope < -0.1:
            signal_type = "SHORT"
            entry_reason = f"Early distribution ({state.spot_cvd_slope:+.2f}), perp not yet participating"
        else:
            return None

        return RegimeSignal(
            timestamp=state.timestamp,
            symbol=state.symbol,
            regime=state.regime,
            signal_type=signal_type,
            confidence=state.confidence,
            entry_reason=entry_reason,
            expected_holding_days=5.0,
            notes="Early entry - best risk/reward. Low leverage (1-3x)."
        )

    @staticmethod
    def _signal_carry(state: RegimeState) -> Optional[RegimeSignal]:
        """Strategy C: Funding carry (delta neutral)"""

        if state.confidence < 0.6:
            return None

        # Check funding profitability
        if abs(state.funding_z) < 1.5:
            return None

        # Determine carry direction
        if state.basis > BASIS_HIGH_THRESHOLD:
            signal_type = "CARRY"
            entry_reason = f"High funding ({state.funding_z:+.2f}), wide basis ({state.basis*100:+.3f}%). Spot long + Perp short."
        elif state.basis < -BASIS_HIGH_THRESHOLD:
            signal_type = "CARRY"
            entry_reason = f"Negative funding ({state.funding_z:+.2f}), negative basis ({state.basis*100:+.3f}%). Spot short + Perp long."
        else:
            return None

        return RegimeSignal(
            timestamp=state.timestamp,
            symbol=state.symbol,
            regime=state.regime,
            signal_type=signal_type,
            confidence=state.confidence,
            entry_reason=entry_reason,
            expected_holding_days=7.0,
            notes="Delta-neutral carry. Monitor funding rate changes."
        )

# ====================================================================
# OPEN INTEREST FETCHER
# ====================================================================

class OIFetcher:
    """Periodically fetch Open Interest"""

    def __init__(self, client: Client, processor: DataProcessor):
        self.client = client
        self.processor = processor
        self.running = False

    async def start(self, symbols: List[str]):
        """Start fetching OI every minute"""
        self.running = True

        while self.running:
            for symbol in symbols:
                try:
                    # Fetch OI
                    oi_data = self.client.futures_open_interest(symbol=symbol)
                    oi = float(oi_data['openInterest'])

                    # Update processor
                    self.processor.update_oi(symbol, oi)

                except Exception as e:
                    logger.error(f"Error fetching OI for {symbol}: {e}")

            # Wait 1 minute
            await asyncio.sleep(60)

    def stop(self):
        self.running = False

# ====================================================================
# WEBSOCKET MANAGER
# ====================================================================

class WebSocketManager:
    """Manage WebSocket connections for spot + perp data"""

    def __init__(
        self,
        processor: DataProcessor,
        telegram: TelegramNotifier,
        stats_logger: StatisticsLogger
    ):
        self.processor = processor
        self.telegram = telegram
        self.stats_logger = stats_logger
        self.symbols: List[str] = []
        self.regime_detector = RegimeDetector()
        self.signal_generator = SignalGenerator()

        # Current regime for each symbol
        self.current_regime: Dict[str, MarketRegime] = {}
        self.regime_start_time: Dict[str, datetime] = {}

        self.running = False
        self.analysis_count = 0

    def set_symbols(self, symbols: List[str]):
        """Set symbols to monitor"""
        self.symbols = symbols
        for symbol in symbols:
            self.processor.init_symbol(symbol)
            self.current_regime[symbol] = MarketRegime.UNKNOWN
            self.regime_start_time[symbol] = datetime.now()

    async def connect_spot_stream(self):
        """Connect to Binance Spot WebSocket"""
        streams = [f"{s.lower()}@trade" for s in self.symbols]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        logger.info(f"📡 Connecting to Spot stream...")

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("✅ Spot stream connected")

                    while self.running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(msg)

                        if 'data' in data:
                            symbol = data['data']['s']
                            self.processor.process_spot_trade(symbol, data['data'])

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Spot stream error: {e}")
                await asyncio.sleep(5)

    async def connect_perp_stream(self):
        """Connect to Binance Futures WebSocket"""
        # Perp trades + funding
        streams = []
        for s in self.symbols:
            streams.append(f"{s.lower()}@trade")
            streams.append(f"{s.lower()}@markPrice")

        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"

        logger.info(f"📡 Connecting to Perp stream...")

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("✅ Perp stream connected")

                    while self.running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(msg)

                        if 'data' not in data:
                            continue

                        event_data = data['data']

                        if event_data.get('e') == 'trade':
                            symbol = event_data['s']
                            self.processor.process_perp_trade(symbol, event_data)

                        elif event_data.get('e') == 'markPriceUpdate':
                            symbol = event_data['s']
                            self.processor.process_funding(symbol, event_data)

                        # Analyze periodically
                        self.analysis_count += 1
                        if self.analysis_count >= 100:
                            await self.analyze_regimes()
                            self.analysis_count = 0

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Perp stream error: {e}")
                await asyncio.sleep(5)

    async def analyze_regimes(self):
        """Analyze current regime for all symbols"""
        for symbol in self.symbols:
            try:
                # Detect regime
                new_state = self.regime_detector.detect_regime(self.processor, symbol)

                if not new_state:
                    continue

                old_regime = self.current_regime.get(symbol, MarketRegime.UNKNOWN)

                # Check regime change
                if new_state.regime != old_regime:
                    logger.info(f"🔄 {symbol}: {old_regime.value} → {new_state.regime.value}")

                    # Calculate duration in old regime
                    duration = (datetime.now() - self.regime_start_time.get(symbol, datetime.now())).total_seconds()

                    # Log regime change
                    self.stats_logger.log_regime(new_state, duration)

                    # Send Telegram notification
                    self.telegram.send_regime_change(old_regime, new_state)

                    # Update state
                    self.current_regime[symbol] = new_state.regime
                    self.regime_start_time[symbol] = datetime.now()

                    # Generate signal
                    signal = self.signal_generator.generate_signal(new_state)
                    if signal:
                        logger.info(f"🚨 SIGNAL: {signal.signal_type} {signal.symbol} (Regime: {signal.regime.value})")
                        self.stats_logger.log_signal(signal)
                        self.telegram.send_signal(signal)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

    async def start(self):
        """Start all streams"""
        self.running = True

        # Start tasks
        tasks = [
            asyncio.create_task(self.connect_spot_stream()),
            asyncio.create_task(self.connect_perp_stream())
        ]

        await asyncio.gather(*tasks)

    def stop(self):
        self.running = False

# ====================================================================
# MAIN SYSTEM
# ====================================================================

class MarketRegimeSystem:
    """Main system orchestrator"""

    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.scanner = SymbolScanner(self.client)
        self.processor = DataProcessor()
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.stats_logger = StatisticsLogger()
        self.ws_manager = WebSocketManager(self.processor, self.telegram, self.stats_logger)
        self.oi_fetcher = OIFetcher(self.client, self.processor)

        self.symbols: List[str] = []
        self.running = False

    async def start(self):
        """Start the system"""
        self.running = True

        logger.info("🚀 Starting Multi-Regime Market Analysis System")
        self.telegram.send_message("🟢 <b>Market Regime System Started</b>\n\nInitializing...")

        # Initial symbol scan
        self.symbols = self.scanner.scan_top_symbols(MAX_SYMBOLS)

        if not self.symbols:
            logger.error("❌ No symbols found!")
            return

        self.ws_manager.set_symbols(self.symbols)

        # Send startup notification
        symbols_str = ", ".join(self.symbols)
        self.telegram.send_message(
            f"✅ <b>System Ready</b>\n\n"
            f"<b>Monitoring {len(self.symbols)} symbols:</b>\n{symbols_str}\n\n"
            f"<b>Regimes:</b>\n"
            f"🟢 Spot-Led Trend\n"
            f"🟡 Perp-Led Euphoria\n"
            f"🟢 Spot Accumulation\n"
            f"🔵 Funding Carry\n\n"
            f"Collecting data..."
        )

        # Start tasks
        tasks = [
            asyncio.create_task(self.ws_manager.start()),
            asyncio.create_task(self.oi_fetcher.start(self.symbols)),
            asyncio.create_task(self.periodic_tasks())
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("⚠️ Keyboard interrupt")
            self.stop()

    async def periodic_tasks(self):
        """Periodic maintenance tasks"""
        while self.running:
            await asyncio.sleep(UPDATE_INTERVAL)

            # Hourly tasks
            logger.info("⏰ Running hourly tasks...")

            # Get statistics
            stats = self.stats_logger.get_regime_statistics(hours=1)

            if stats:
                # Calculate time distribution
                total_time = sum(stats.get('total_time_by_regime', {}).values())
                time_dist = {}

                if total_time > 0:
                    for regime, seconds in stats.get('total_time_by_regime', {}).items():
                        time_dist[regime] = (seconds / total_time) * 100

                # Top symbols
                top_symbols = list(stats.get('symbol_counts', {}).items())[:5]

                # Send summary
                self.telegram.send_hourly_summary({
                    'time_distribution': time_dist,
                    'signals_count': stats.get('total_samples', 0),
                    'top_symbols': top_symbols
                })

    def stop(self):
        """Stop the system"""
        logger.info("🛑 Stopping system...")
        self.running = False
        self.ws_manager.stop()
        self.oi_fetcher.stop()
        self.telegram.send_message("🔴 <b>System Stopped</b>")

# ====================================================================
# MAIN ENTRY POINT
# ====================================================================

async def main():
    """Main entry point"""
    system = MarketRegimeSystem()

    try:
        await system.start()
    except KeyboardInterrupt:
        system.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        system.telegram.send_message(f"🔴 <b>System Error</b>\n\n{str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
