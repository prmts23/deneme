"""
Backtest Engine for ABCD Pattern Trading Strategy
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

import sys
sys.path.append('..')
from abcd_pattern.detector import ABCDDetector
from abcd_pattern.config import BacktestConfig

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Trading position"""
    entry_time: datetime
    entry_price: float
    size: float
    side: PositionSide
    stop_loss: float
    take_profit: float
    confidence: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class BacktestResult:
    """Backtest results container"""
    positions: List[Position] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    total_return_pct: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    avg_trade_duration: float = 0.0  # in hours

    initial_capital: float = 0.0
    final_capital: float = 0.0


class ABCDBacktest:
    """Backtest engine for ABCD pattern trading"""

    def __init__(
        self,
        detector: ABCDDetector,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtest engine

        Args:
            detector: ABCD pattern detector
            config: Backtest configuration
        """
        self.detector = detector
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self) -> None:
        """Reset backtest state"""
        self.capital = self.config.initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on capital

        Args:
            price: Current price

        Returns:
            Position size in base currency
        """
        position_value = self.capital * self.config.position_size_pct
        size = position_value / price
        return size

    def open_position(
        self,
        timestamp: datetime,
        price: float,
        confidence: float,
        side: PositionSide = PositionSide.LONG
    ) -> Optional[Position]:
        """
        Open new position

        Args:
            timestamp: Entry timestamp
            price: Entry price
            confidence: Pattern confidence
            side: Position side

        Returns:
            Position object or None if max positions reached
        """
        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            logger.debug(f"Max positions reached ({self.config.max_positions})")
            return None

        # Calculate position size
        size = self.calculate_position_size(price)

        # Calculate stop loss and take profit
        if side == PositionSide.LONG:
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:  # SHORT
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)

        position = Position(
            entry_time=timestamp,
            entry_price=price,
            size=size,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence
        )

        self.positions.append(position)

        # Deduct commission
        commission = size * price * self.config.commission_pct
        self.capital -= commission

        logger.info(f"Opened {side.value} position: {size:.4f} @ {price:.2f}, "
                   f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

        return position

    def close_position(
        self,
        position: Position,
        timestamp: datetime,
        price: float,
        reason: str
    ) -> None:
        """
        Close existing position

        Args:
            position: Position to close
            timestamp: Exit timestamp
            price: Exit price
            reason: Exit reason
        """
        position.exit_time = timestamp
        position.exit_price = price
        position.exit_reason = reason

        # Calculate PnL
        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - price) * position.size

        # Deduct commission
        commission = position.size * price * self.config.commission_pct
        pnl -= commission

        position.pnl = pnl
        position.pnl_pct = (pnl / (position.entry_price * position.size)) * 100

        # Update capital
        self.capital += pnl

        # Move to closed positions
        self.positions.remove(position)
        self.closed_positions.append(position)

        logger.info(f"Closed {position.side.value} position: PnL {pnl:.2f} "
                   f"({position.pnl_pct:.2f}%), Reason: {reason}")

    def check_exits(
        self,
        timestamp: datetime,
        current_bar: pd.Series
    ) -> None:
        """
        Check if any positions should be closed

        Args:
            timestamp: Current timestamp
            current_bar: Current OHLCV bar
        """
        positions_to_close = []

        for position in self.positions:
            if position.side == PositionSide.LONG:
                # Check stop loss
                if current_bar['Low'] <= position.stop_loss:
                    positions_to_close.append((position, position.stop_loss, "Stop Loss"))
                # Check take profit
                elif current_bar['High'] >= position.take_profit:
                    positions_to_close.append((position, position.take_profit, "Take Profit"))

            else:  # SHORT
                # Check stop loss
                if current_bar['High'] >= position.stop_loss:
                    positions_to_close.append((position, position.stop_loss, "Stop Loss"))
                # Check take profit
                elif current_bar['Low'] <= position.take_profit:
                    positions_to_close.append((position, position.take_profit, "Take Profit"))

        # Close positions
        for position, exit_price, reason in positions_to_close:
            self.close_position(position, timestamp, exit_price, reason)

    def run(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        check_interval: int = 5
    ) -> BacktestResult:
        """
        Run backtest on OHLCV data

        Args:
            df: OHLCV DataFrame
            lookback: Lookback period for pattern detection
            check_interval: Check for patterns every N bars

        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest on {len(df)} bars...")
        self.reset()

        # Ensure we have enough data
        if len(df) < lookback:
            raise ValueError(f"Not enough data: {len(df)} < {lookback}")

        # Main backtest loop
        for i in range(lookback, len(df), check_interval):
            current_time = df.index[i]
            current_bar = df.iloc[i]

            # Check exits first
            self.check_exits(current_time, current_bar)

            # Look for new patterns
            if len(self.positions) < self.config.max_positions:
                window_df = df.iloc[i-lookback:i]

                try:
                    confidence, is_pattern = self.detector.detect_pattern(window_df)

                    if is_pattern:
                        # Open position
                        # Assuming bullish pattern = LONG
                        # In real scenario, you'd determine direction from pattern
                        self.open_position(
                            timestamp=current_time,
                            price=current_bar['Close'],
                            confidence=confidence,
                            side=PositionSide.LONG
                        )

                except Exception as e:
                    logger.error(f"Error detecting pattern at {current_time}: {e}")
                    continue

            # Record equity
            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            for position in self.positions:
                if position.side == PositionSide.LONG:
                    unrealized_pnl += (current_bar['Close'] - position.entry_price) * position.size
                else:
                    unrealized_pnl += (position.entry_price - current_bar['Close']) * position.size

            current_equity = self.capital + unrealized_pnl
            self.equity_curve.append(current_equity)
            self.timestamps.append(current_time)

        # Close all remaining positions at end
        if self.positions:
            final_bar = df.iloc[-1]
            final_time = df.index[-1]
            for position in list(self.positions):
                self.close_position(
                    position,
                    final_time,
                    final_bar['Close'],
                    "End of Backtest"
                )

        # Calculate results
        result = self.calculate_results()

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Win rate: {result.win_rate:.2f}%, "
                   f"Total return: {result.total_return_pct:.2f}%")

        return result

    def calculate_results(self) -> BacktestResult:
        """
        Calculate backtest performance metrics

        Returns:
            BacktestResult object
        """
        result = BacktestResult()
        result.positions = self.closed_positions
        result.equity_curve = self.equity_curve
        result.timestamps = self.timestamps

        result.initial_capital = self.config.initial_capital
        result.final_capital = self.capital

        if not self.closed_positions:
            logger.warning("No closed positions to analyze")
            return result

        # Basic metrics
        result.total_trades = len(self.closed_positions)

        winning_positions = [p for p in self.closed_positions if p.pnl > 0]
        losing_positions = [p for p in self.closed_positions if p.pnl <= 0]

        result.winning_trades = len(winning_positions)
        result.losing_trades = len(losing_positions)
        result.win_rate = (result.winning_trades / result.total_trades) * 100

        # PnL metrics
        result.total_pnl = sum(p.pnl for p in self.closed_positions)
        result.total_return_pct = (result.total_pnl / self.config.initial_capital) * 100

        if winning_positions:
            result.avg_win = np.mean([p.pnl for p in winning_positions])
            result.largest_win = max(p.pnl for p in winning_positions)

        if losing_positions:
            result.avg_loss = np.mean([p.pnl for p in losing_positions])
            result.largest_loss = min(p.pnl for p in losing_positions)

        # Profit factor
        total_wins = sum(p.pnl for p in winning_positions) if winning_positions else 0
        total_losses = abs(sum(p.pnl for p in losing_positions)) if losing_positions else 1
        result.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Drawdown
        if self.equity_curve:
            equity_array = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = running_max - equity_array
            result.max_drawdown = float(np.max(drawdown))
            result.max_drawdown_pct = (result.max_drawdown / self.config.initial_capital) * 100

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

        # Average trade duration
        durations = []
        for p in self.closed_positions:
            if p.exit_time and p.entry_time:
                duration = (p.exit_time - p.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)

        if durations:
            result.avg_trade_duration = np.mean(durations)

        return result

    def print_results(self, result: BacktestResult) -> None:
        """
        Print backtest results

        Args:
            result: BacktestResult object
        """
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        print(f"\nCapital:")
        print(f"  Initial: ${result.initial_capital:,.2f}")
        print(f"  Final:   ${result.final_capital:,.2f}")
        print(f"  Total PnL: ${result.total_pnl:,.2f} ({result.total_return_pct:.2f}%)")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Winning: {result.winning_trades} ({result.win_rate:.2f}%)")
        print(f"  Losing: {result.losing_trades}")

        print(f"\nPnL Metrics:")
        print(f"  Avg Win: ${result.avg_win:.2f}")
        print(f"  Avg Loss: ${result.avg_loss:.2f}")
        print(f"  Largest Win: ${result.largest_win:.2f}")
        print(f"  Largest Loss: ${result.largest_loss:.2f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")

        print(f"\nOther:")
        print(f"  Avg Trade Duration: {result.avg_trade_duration:.1f} hours")

        print("\n" + "="*60)

    def plot_results(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot backtest results

        Args:
            result: BacktestResult object
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Equity curve
        axes[0].plot(result.timestamps, result.equity_curve, linewidth=2)
        axes[0].axhline(
            y=result.initial_capital,
            color='gray',
            linestyle='--',
            label='Initial Capital'
        )
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Trade PnL distribution
        if result.positions:
            pnls = [p.pnl for p in result.positions]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]

            axes[1].bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Trade #')
            axes[1].set_ylabel('PnL ($)')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
