#!/usr/bin/env python3
"""
Script to run backtest with ABCD pattern detector
"""
import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abcd_pattern import (
    ABCDDetector,
    BacktestConfig,
    load_ohlcv_data,
    setup_logging,
    save_results_to_json
)
from backtest import ABCDBacktest


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run backtest with ABCD pattern detector'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/abcd_detector.h5',
        help='Path to trained model'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to OHLCV CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='backtest_results',
        help='Output directory for results'
    )

    # Backtest parameters
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000.0,
        help='Initial capital'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Position size as percentage of capital (0.1 = 10%%)'
    )

    parser.add_argument(
        '--max-positions',
        type=int,
        default=3,
        help='Maximum concurrent positions'
    )

    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.02,
        help='Stop loss percentage (0.02 = 2%%)'
    )

    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.04,
        help='Take profit percentage (0.04 = 4%%)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Detection confidence threshold (0.7 = 70%%)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission percentage (0.001 = 0.1%%)'
    )

    # Detection parameters
    parser.add_argument(
        '--lookback',
        type=int,
        default=20,
        help='Lookback period for pattern detection'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=5,
        help='Check for patterns every N bars'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("ABCD Pattern Backtest")
    print("="*60)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    detector = ABCDDetector(model_path=args.model_path)

    # Load data
    print(f"Loading OHLCV data from {args.data_path}...")
    df = load_ohlcv_data(args.data_path)
    print(f"  Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        position_size_pct=args.position_size,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        detection_threshold=args.threshold,
        commission_pct=args.commission
    )

    print("\nBacktest Configuration:")
    print(f"  Initial capital: ${config.initial_capital:,.2f}")
    print(f"  Position size: {config.position_size_pct*100:.1f}%")
    print(f"  Max positions: {config.max_positions}")
    print(f"  Stop loss: {config.stop_loss_pct*100:.1f}%")
    print(f"  Take profit: {config.take_profit_pct*100:.1f}%")
    print(f"  Detection threshold: {config.detection_threshold*100:.1f}%")
    print(f"  Commission: {config.commission_pct*100:.2f}%")

    print("\nDetection Parameters:")
    print(f"  Lookback period: {args.lookback} bars")
    print(f"  Check interval: {args.check_interval} bars")

    print("\n" + "="*60)
    print("Running Backtest...")
    print("="*60 + "\n")

    # Run backtest
    start_time = datetime.now()

    backtest = ABCDBacktest(detector, config)
    result = backtest.run(
        df,
        lookback=args.lookback,
        check_interval=args.check_interval
    )

    end_time = datetime.now()
    backtest_time = (end_time - start_time).total_seconds()

    # Print results
    backtest.print_results(result)

    print(f"\nBacktest execution time: {backtest_time:.1f} seconds")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save plot
    plot_path = os.path.join(args.output_dir, f"backtest_{timestamp}.png")
    backtest.plot_results(result, save_path=plot_path)
    print(f"\nResults plot saved to: {plot_path}")

    # Save metrics to JSON
    metrics_dict = {
        'backtest_config': {
            'initial_capital': config.initial_capital,
            'position_size_pct': config.position_size_pct,
            'max_positions': config.max_positions,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct,
            'detection_threshold': config.detection_threshold,
            'commission_pct': config.commission_pct
        },
        'detection_params': {
            'lookback': args.lookback,
            'check_interval': args.check_interval
        },
        'data_info': {
            'data_path': args.data_path,
            'start_date': str(df.index[0]),
            'end_date': str(df.index[-1]),
            'total_bars': len(df)
        },
        'results': {
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'total_pnl': result.total_pnl,
            'total_return_pct': result.total_return_pct,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_pct': result.max_drawdown_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'profit_factor': result.profit_factor,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'largest_win': result.largest_win,
            'largest_loss': result.largest_loss,
            'avg_trade_duration': result.avg_trade_duration,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital
        },
        'trades': [
            {
                'entry_time': str(pos.entry_time),
                'entry_price': pos.entry_price,
                'exit_time': str(pos.exit_time) if pos.exit_time else None,
                'exit_price': pos.exit_price,
                'size': pos.size,
                'side': pos.side.value,
                'pnl': pos.pnl,
                'pnl_pct': pos.pnl_pct,
                'exit_reason': pos.exit_reason,
                'confidence': pos.confidence
            }
            for pos in result.positions
        ]
    }

    json_path = os.path.join(args.output_dir, f"backtest_{timestamp}.json")
    save_results_to_json(metrics_dict, json_path)
    print(f"Results JSON saved to: {json_path}")

    print("\n✓ Backtest complete!")


if __name__ == '__main__':
    main()
