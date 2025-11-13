"""
Regime Performance Analyzer
===========================

Analyze which market regimes are most profitable

Usage:
    python regime_analyzer.py
    python regime_analyzer.py --days 7
    python regime_analyzer.py --regime SPOT_LED_TREND
"""

import pandas as pd
import argparse
from datetime import datetime, timedelta
import sys

# ====================================================================
# CONFIGURATION
# ====================================================================

REGIME_LOG_FILE = "regime_stats.csv"
SIGNAL_LOG_FILE = "regime_signals.csv"

# ====================================================================
# DATA LOADER
# ====================================================================

def load_regime_data(file_path: str, days: int = None) -> pd.DataFrame:
    """Load regime statistics"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]

        print(f"✅ Loaded {len(df)} regime samples")
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def load_signal_data(file_path: str, days: int = None) -> pd.DataFrame:
    """Load signal data"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]

        print(f"✅ Loaded {len(df)} signals")
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

# ====================================================================
# ANALYSIS FUNCTIONS
# ====================================================================

def analyze_regime_distribution(regime_df: pd.DataFrame):
    """Analyze time spent in each regime"""
    print("\n" + "="*80)
    print("📊 REGIME TIME DISTRIBUTION")
    print("="*80)

    total_time = regime_df['duration_seconds'].sum()

    regime_time = regime_df.groupby('regime').agg({
        'duration_seconds': ['sum', 'count', 'mean']
    })

    regime_time.columns = ['Total_Seconds', 'Occurrences', 'Avg_Duration']
    regime_time['Total_Hours'] = regime_time['Total_Seconds'] / 3600
    regime_time['Percentage'] = (regime_time['Total_Seconds'] / total_time) * 100
    regime_time = regime_time.sort_values('Total_Seconds', ascending=False)

    print(regime_time.round(2))

    print(f"\n📈 INSIGHTS:")
    top_regime = regime_time.index[0]
    top_pct = regime_time.iloc[0]['Percentage']
    print(f"  • Most common regime: {top_regime} ({top_pct:.1f}% of time)")
    print(f"  • Total monitored time: {total_time/3600:.1f} hours")

def analyze_regime_metrics(regime_df: pd.DataFrame):
    """Analyze metrics by regime"""
    print("\n" + "="*80)
    print("📈 REGIME METRICS ANALYSIS")
    print("="*80)

    metrics = regime_df.groupby('regime').agg({
        'confidence': ['mean', 'std'],
        'spot_cvd_slope': ['mean', 'std'],
        'perp_cvd_slope': ['mean', 'std'],
        'oi_change_pct': ['mean', 'std'],
        'funding_z': ['mean', 'std'],
        'basis': ['mean', 'std']
    }).round(4)

    print(metrics)

def analyze_signal_performance(signal_df: pd.DataFrame):
    """Analyze signals by regime"""
    print("\n" + "="*80)
    print("🚨 SIGNAL ANALYSIS BY REGIME")
    print("="*80)

    if signal_df.empty:
        print("  No signals generated yet")
        return

    # Count by regime
    signal_counts = signal_df.groupby('regime').agg({
        'signal_type': 'count',
        'confidence': 'mean',
        'expected_holding_days': 'mean'
    })

    signal_counts.columns = ['Count', 'Avg_Confidence', 'Avg_Hold_Days']
    signal_counts = signal_counts.sort_values('Count', ascending=False)

    print(signal_counts.round(2))

    # Signal types within each regime
    print("\n📊 SIGNAL TYPES BY REGIME:")
    for regime in signal_df['regime'].unique():
        regime_signals = signal_df[signal_df['regime'] == regime]
        type_counts = regime_signals['signal_type'].value_counts()
        print(f"\n  {regime}:")
        for sig_type, count in type_counts.items():
            print(f"    • {sig_type}: {count}")

def analyze_by_symbol(regime_df: pd.DataFrame, signal_df: pd.DataFrame):
    """Analyze by symbol"""
    print("\n" + "="*80)
    print("💰 ANALYSIS BY SYMBOL")
    print("="*80)

    # Regime changes per symbol
    symbol_regime_changes = regime_df.groupby('symbol').agg({
        'regime': 'count',
        'confidence': 'mean',
        'duration_seconds': 'sum'
    })

    symbol_regime_changes.columns = ['Regime_Changes', 'Avg_Confidence', 'Total_Time_Seconds']
    symbol_regime_changes['Total_Time_Hours'] = symbol_regime_changes['Total_Time_Seconds'] / 3600
    symbol_regime_changes = symbol_regime_changes.sort_values('Regime_Changes', ascending=False)

    print("\n📊 Top Symbols by Regime Activity:")
    print(symbol_regime_changes.head(10).round(2))

    if not signal_df.empty:
        # Signals per symbol
        symbol_signals = signal_df['symbol'].value_counts()

        print("\n🚨 Signals by Symbol:")
        print(symbol_signals.head(10))

def analyze_regime_transitions(regime_df: pd.DataFrame):
    """Analyze regime transitions"""
    print("\n" + "="*80)
    print("🔄 REGIME TRANSITION ANALYSIS")
    print("="*80)

    # Sort by time
    regime_df = regime_df.sort_values(['symbol', 'timestamp'])

    # Calculate transitions
    transitions = {}

    for symbol in regime_df['symbol'].unique():
        symbol_data = regime_df[regime_df['symbol'] == symbol].reset_index(drop=True)

        for i in range(len(symbol_data) - 1):
            from_regime = symbol_data.loc[i, 'regime']
            to_regime = symbol_data.loc[i + 1, 'regime']

            transition = f"{from_regime} → {to_regime}"

            if transition not in transitions:
                transitions[transition] = 0

            transitions[transition] += 1

    # Sort by frequency
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)

    print("\n📊 Top 10 Regime Transitions:")
    for transition, count in sorted_transitions[:10]:
        print(f"  • {transition}: {count} times")

def analyze_best_regime(regime_df: pd.DataFrame, signal_df: pd.DataFrame):
    """Determine which regime is best"""
    print("\n" + "="*80)
    print("🏆 BEST REGIME ANALYSIS")
    print("="*80)

    # Metrics to consider:
    # 1. Signal frequency
    # 2. Average confidence
    # 3. Time spent in regime

    regime_scores = {}

    for regime in regime_df['regime'].unique():
        regime_data = regime_df[regime_df['regime'] == regime]

        # Time spent
        time_pct = regime_data['duration_seconds'].sum() / regime_df['duration_seconds'].sum()

        # Confidence
        avg_confidence = regime_data['confidence'].mean()

        # Signals
        regime_signals = signal_df[signal_df['regime'] == regime] if not signal_df.empty else pd.DataFrame()
        signal_count = len(regime_signals)
        signal_frequency = signal_count / (regime_data['duration_seconds'].sum() / 3600) if regime_data['duration_seconds'].sum() > 0 else 0

        regime_scores[regime] = {
            'time_pct': time_pct * 100,
            'avg_confidence': avg_confidence,
            'signal_count': signal_count,
            'signals_per_hour': signal_frequency
        }

    # Create DataFrame
    scores_df = pd.DataFrame(regime_scores).T
    scores_df = scores_df.sort_values('signals_per_hour', ascending=False)

    print("\n📊 Regime Scoring:")
    print(scores_df.round(4))

    print("\n💡 RECOMMENDATIONS:")
    best_by_signals = scores_df.index[0]
    print(f"  ✅ Most active regime: {best_by_signals}")
    print(f"     → Generates {scores_df.loc[best_by_signals, 'signals_per_hour']:.2f} signals/hour")

    most_common = scores_df.sort_values('time_pct', ascending=False).index[0]
    print(f"  ✅ Most common regime: {most_common}")
    print(f"     → Occurs {scores_df.loc[most_common, 'time_pct']:.1f}% of the time")

    highest_confidence = scores_df.sort_values('avg_confidence', ascending=False).index[0]
    print(f"  ✅ Highest confidence regime: {highest_confidence}")
    print(f"     → Average confidence: {scores_df.loc[highest_confidence, 'avg_confidence']:.1%}")

# ====================================================================
# MAIN REPORT
# ====================================================================

def generate_report(regime_df: pd.DataFrame, signal_df: pd.DataFrame):
    """Generate comprehensive report"""

    print("\n" + "="*80)
    print("📊 MULTI-REGIME MARKET ANALYSIS REPORT")
    print("="*80)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Period: {regime_df['timestamp'].min().date()} to {regime_df['timestamp'].max().date()}")
    print(f"Total Samples: {len(regime_df)}")
    print(f"Total Signals: {len(signal_df)}")
    print("="*80)

    # Run analyses
    analyze_regime_distribution(regime_df)
    analyze_regime_metrics(regime_df)
    analyze_signal_performance(signal_df)
    analyze_by_symbol(regime_df, signal_df)
    analyze_regime_transitions(regime_df)
    analyze_best_regime(regime_df, signal_df)

    print("\n" + "="*80)
    print("📝 NEXT STEPS:")
    print("="*80)
    print("  1. Let system collect data for at least 24-48 hours")
    print("  2. Focus on regimes with highest signal frequency")
    print("  3. Adjust thresholds based on regime performance")
    print("  4. Track actual trade results for each regime")
    print("  5. Iterate and optimize")
    print("\n")

# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze regime performance')
    parser.add_argument('--days', type=int, help='Analyze last N days')
    parser.add_argument('--regime', type=str, help='Filter by specific regime')

    args = parser.parse_args()

    # Load data
    regime_df = load_regime_data(REGIME_LOG_FILE, args.days)
    signal_df = load_signal_data(SIGNAL_LOG_FILE, args.days)

    if regime_df.empty:
        print("❌ No regime data found. Run the system first!")
        return

    # Filter by regime if specified
    if args.regime:
        regime_df = regime_df[regime_df['regime'] == args.regime]
        signal_df = signal_df[signal_df['regime'] == args.regime]
        print(f"\n🔍 Filtering by regime: {args.regime}")

    # Generate report
    generate_report(regime_df, signal_df)

if __name__ == "__main__":
    main()
