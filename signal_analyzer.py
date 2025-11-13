"""
Signal Performance Analyzer
============================

Analyzes signal_log.csv to provide insights:
- Win rate
- Profit/Loss by signal type
- Best performing conditions
- Time-based analysis
- Symbol-based analysis

Usage:
    python signal_analyzer.py
    python signal_analyzer.py --file custom_log.csv
    python signal_analyzer.py --days 7  # Last 7 days only
"""

import pandas as pd
import argparse
from datetime import datetime, timedelta
from typing import Dict, List
import sys

# ====================================================================
# CONFIGURATION
# ====================================================================

DEFAULT_LOG_FILE = "signals_log.csv"
HOLD_PERIOD_DAYS = 3  # Assumed hold period for analysis

# ====================================================================
# DATA LOADER
# ====================================================================

def load_signals(file_path: str, days: int = None) -> pd.DataFrame:
    """Load signals from CSV"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by date if specified
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]

        print(f"✅ Loaded {len(df)} signals from {file_path}")
        if days:
            print(f"   (Last {days} days)")

        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        print(f"   Make sure the system has generated some signals first.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)

# ====================================================================
# ANALYSIS FUNCTIONS
# ====================================================================

def analyze_overview(df: pd.DataFrame) -> Dict:
    """Overall statistics"""
    if df.empty:
        return {}

    total_signals = len(df)
    unique_symbols = df['symbol'].nunique()
    date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"

    signal_types = df['signal_type'].value_counts().to_dict()

    avg_funding = df['funding_rate'].mean()
    avg_confidence = df['confidence'].mean()
    avg_payback = df['payback_days'].mean()
    avg_daily_profit = df['expected_daily_profit'].mean()

    return {
        'total_signals': total_signals,
        'unique_symbols': unique_symbols,
        'date_range': date_range,
        'signal_types': signal_types,
        'avg_funding': avg_funding,
        'avg_confidence': avg_confidence,
        'avg_payback': avg_payback,
        'avg_daily_profit': avg_daily_profit
    }

def analyze_by_signal_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analysis grouped by signal type (LONG/SHORT)"""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby('signal_type').agg({
        'symbol': 'count',
        'confidence': 'mean',
        'funding_rate': 'mean',
        'cvd_change_15m': 'mean',
        'expected_daily_profit': 'mean',
        'payback_days': 'mean'
    }).round(4)

    grouped.columns = ['Count', 'Avg_Confidence', 'Avg_Funding', 'Avg_CVD', 'Avg_Daily_Profit', 'Avg_Payback']

    return grouped

def analyze_by_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Analysis grouped by signal reason"""
    if df.empty or 'notes' not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby('notes').agg({
        'symbol': 'count',
        'confidence': 'mean',
        'funding_rate': 'mean',
        'expected_daily_profit': 'mean',
        'payback_days': 'mean'
    }).round(4)

    grouped.columns = ['Count', 'Avg_Confidence', 'Avg_Funding', 'Avg_Daily_Profit', 'Avg_Payback']

    # Sort by count
    grouped = grouped.sort_values('Count', ascending=False)

    return grouped

def analyze_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Analysis grouped by symbol"""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby('symbol').agg({
        'signal_type': 'count',
        'confidence': 'mean',
        'funding_rate': 'mean',
        'expected_daily_profit': 'sum',
        'payback_days': 'mean'
    }).round(4)

    grouped.columns = ['Signal_Count', 'Avg_Confidence', 'Avg_Funding', 'Total_Expected_Profit', 'Avg_Payback']

    # Sort by signal count
    grouped = grouped.sort_values('Signal_Count', ascending=False)

    return grouped

def analyze_by_cvd_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Analysis grouped by CVD direction"""
    if df.empty or 'cvd_direction' not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby('cvd_direction').agg({
        'symbol': 'count',
        'signal_type': lambda x: x.value_counts().to_dict(),
        'confidence': 'mean',
        'funding_rate': 'mean',
        'expected_daily_profit': 'mean'
    }).round(4)

    grouped.columns = ['Count', 'Signal_Types', 'Avg_Confidence', 'Avg_Funding', 'Avg_Daily_Profit']

    return grouped

def analyze_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Analysis by hour of day"""
    if df.empty:
        return pd.DataFrame()

    df['hour'] = df['timestamp'].dt.hour

    grouped = df.groupby('hour').agg({
        'symbol': 'count',
        'confidence': 'mean',
        'funding_rate': 'mean',
        'expected_daily_profit': 'mean'
    }).round(4)

    grouped.columns = ['Signal_Count', 'Avg_Confidence', 'Avg_Funding', 'Avg_Daily_Profit']

    return grouped

def analyze_profitability_estimate(df: pd.DataFrame, hold_days: int = HOLD_PERIOD_DAYS) -> Dict:
    """
    Estimate profitability if all signals were taken

    Note: This is THEORETICAL - actual results depend on execution and market conditions
    """
    if df.empty:
        return {}

    # Expected profit per signal (assuming hold_days duration)
    df['estimated_profit'] = df['expected_daily_profit'] * hold_days - df['fees_total']

    total_estimated_profit = df['estimated_profit'].sum()
    avg_profit_per_signal = df['estimated_profit'].mean()
    profitable_signals = len(df[df['estimated_profit'] > 0])
    unprofitable_signals = len(df[df['estimated_profit'] <= 0])

    # Theoretical win rate (signals with positive expected profit)
    theoretical_win_rate = profitable_signals / len(df) if len(df) > 0 else 0

    return {
        'hold_days': hold_days,
        'total_signals': len(df),
        'profitable_signals': profitable_signals,
        'unprofitable_signals': unprofitable_signals,
        'theoretical_win_rate': theoretical_win_rate,
        'total_estimated_profit': total_estimated_profit,
        'avg_profit_per_signal': avg_profit_per_signal,
        'best_signal_profit': df['estimated_profit'].max(),
        'worst_signal_profit': df['estimated_profit'].min()
    }

# ====================================================================
# REPORT GENERATOR
# ====================================================================

def print_report(df: pd.DataFrame):
    """Print comprehensive analysis report"""

    print("\n" + "="*80)
    print("📊 SIGNAL PERFORMANCE ANALYSIS")
    print("="*80)

    # Overview
    overview = analyze_overview(df)
    if overview:
        print("\n📈 OVERVIEW:")
        print(f"  Total Signals: {overview['total_signals']}")
        print(f"  Unique Symbols: {overview['unique_symbols']}")
        print(f"  Date Range: {overview['date_range']}")
        print(f"  Signal Types: {overview['signal_types']}")
        print(f"\n  Average Metrics:")
        print(f"    Funding Rate: {overview['avg_funding']*100:.4f}%")
        print(f"    Confidence: {overview['avg_confidence']:.2%}")
        print(f"    Payback Days: {overview['avg_payback']:.2f}")
        print(f"    Daily Profit: ${overview['avg_daily_profit']:.2f}")

    # By Signal Type
    print("\n" + "-"*80)
    print("📊 BY SIGNAL TYPE (LONG vs SHORT):")
    by_type = analyze_by_signal_type(df)
    if not by_type.empty:
        print(by_type.to_string())
    else:
        print("  No data")

    # By Signal Reason
    print("\n" + "-"*80)
    print("🎯 BY SIGNAL REASON:")
    by_reason = analyze_by_reason(df)
    if not by_reason.empty:
        print(by_reason.to_string())
    else:
        print("  No data")

    # By Symbol
    print("\n" + "-"*80)
    print("💰 BY SYMBOL (Top 10):")
    by_symbol = analyze_by_symbol(df)
    if not by_symbol.empty:
        print(by_symbol.head(10).to_string())
    else:
        print("  No data")

    # By CVD Direction
    print("\n" + "-"*80)
    print("📈 BY CVD DIRECTION:")
    by_cvd = analyze_by_cvd_direction(df)
    if not by_cvd.empty:
        print(by_cvd.to_string())
    else:
        print("  No data")

    # By Hour
    print("\n" + "-"*80)
    print("⏰ BY HOUR OF DAY:")
    by_hour = analyze_by_hour(df)
    if not by_hour.empty:
        print(by_hour.to_string())
    else:
        print("  No data")

    # Profitability Estimate
    print("\n" + "-"*80)
    print(f"💵 PROFITABILITY ESTIMATE ({HOLD_PERIOD_DAYS}-day hold):")
    print("   ⚠️  NOTE: This is THEORETICAL based on expected funding rates")
    print("   ⚠️  Actual results depend on execution and market conditions\n")

    profitability = analyze_profitability_estimate(df, HOLD_PERIOD_DAYS)
    if profitability:
        print(f"  Total Signals: {profitability['total_signals']}")
        print(f"  Profitable: {profitability['profitable_signals']} ({profitability['theoretical_win_rate']:.1%})")
        print(f"  Unprofitable: {profitability['unprofitable_signals']}")
        print(f"\n  Total Est. Profit: ${profitability['total_estimated_profit']:.2f}")
        print(f"  Avg per Signal: ${profitability['avg_profit_per_signal']:.2f}")
        print(f"  Best Signal: ${profitability['best_signal_profit']:.2f}")
        print(f"  Worst Signal: ${profitability['worst_signal_profit']:.2f}")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS:")
    print("="*80)

    if not by_reason.empty:
        best_reason = by_reason.sort_values('Avg_Daily_Profit', ascending=False).index[0]
        print(f"  ✅ Best Signal Type: {best_reason}")
        print(f"     → Focus on this condition for highest expected returns")

    if not by_symbol.empty:
        best_symbol = by_symbol.sort_values('Total_Expected_Profit', ascending=False).index[0]
        print(f"  ✅ Most Profitable Symbol: {best_symbol}")
        print(f"     → This symbol generated most cumulative signals")

    if not by_hour.empty:
        best_hour = by_hour.sort_values('Signal_Count', ascending=False).index[0]
        print(f"  ✅ Most Active Hour: {best_hour}:00 UTC")
        print(f"     → Most signals occur at this time")

    print("\n" + "="*80)
    print("📝 NEXT STEPS:")
    print("="*80)
    print("  1. Collect more data (at least 1-2 weeks for reliable statistics)")
    print("  2. Track actual executed trades vs expected profit")
    print("  3. Compare different signal reasons (HIGH_FUNDING vs EXTREME_SPIKE)")
    print("  4. Adjust FUNDING_Z_THRESHOLD based on what works best")
    print("  5. Consider filtering out low-confidence signals (< 50%)")
    print("\n")

# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze trading signal performance')
    parser.add_argument('--file', type=str, default=DEFAULT_LOG_FILE, help='Path to signal log CSV')
    parser.add_argument('--days', type=int, help='Analyze only last N days')
    parser.add_argument('--export', type=str, help='Export analysis to file')

    args = parser.parse_args()

    # Load data
    df = load_signals(args.file, args.days)

    if df.empty:
        print("❌ No signals found in log file")
        print("   Run the system first to collect data")
        return

    # Print report
    print_report(df)

    # Export if requested
    if args.export:
        try:
            # Export summary statistics
            overview = analyze_overview(df)
            by_type = analyze_by_signal_type(df)
            by_reason = analyze_by_reason(df)
            by_symbol = analyze_by_symbol(df)

            with open(args.export, 'w') as f:
                f.write("SIGNAL ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Data Range: {overview.get('date_range', 'N/A')}\n\n")

                f.write("BY SIGNAL TYPE:\n")
                f.write(by_type.to_string() + "\n\n")

                f.write("BY SIGNAL REASON:\n")
                f.write(by_reason.to_string() + "\n\n")

                f.write("BY SYMBOL:\n")
                f.write(by_symbol.to_string() + "\n")

            print(f"✅ Analysis exported to: {args.export}")
        except Exception as e:
            print(f"❌ Error exporting: {e}")

if __name__ == "__main__":
    main()
