"""
Quick Test Script - Debug Data Loading and Feature Engineering

Run this to diagnose issues before full training
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML TRADING SYSTEM - QUICK DIAGNOSTIC TEST")
print("=" * 80 + "\n")

# ============================================================
# TEST 1: Config and Data Path
# ============================================================
print("TEST 1: Checking configuration...")

try:
    import config
    print(f"✓ Config loaded successfully")
    print(f"  DATA_PATH: {config.DATA_PATH}")
    print(f"  Symbol: {config.SYMBOL}")
    print(f"  Timeframe: {config.TIMEFRAME}")
except Exception as e:
    print(f"❌ Error loading config: {e}")
    print("   Make sure config.py exists and is valid")
    exit(1)

print()

# ============================================================
# TEST 2: Data File Exists
# ============================================================
print("TEST 2: Checking data file...")

if not os.path.exists(config.DATA_PATH):
    print(f"❌ Data file not found: {config.DATA_PATH}")
    print("\n⚠️  SOLUTION:")
    print("   1. Update DATA_PATH in config.py")
    print("   2. Make sure the CSV file exists")
    print("   3. Use absolute path (not relative)")
    exit(1)

print(f"✓ Data file exists: {config.DATA_PATH}")
file_size_mb = os.path.getsize(config.DATA_PATH) / (1024 * 1024)
print(f"  File size: {file_size_mb:.2f} MB")
print()

# ============================================================
# TEST 3: Load Data
# ============================================================
print("TEST 3: Loading data...")

try:
    df = pd.read_csv(config.DATA_PATH)
    print(f"✓ Data loaded successfully")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

print()

# ============================================================
# TEST 4: Check Required Columns
# ============================================================
print("TEST 4: Checking required columns...")

# Standardize column names
df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'Timestamp': 'timestamp'
}, inplace=True)

required_cols = ['open', 'high', 'low', 'close', 'volume']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"❌ Missing required columns: {missing_cols}")
    print(f"   Available columns: {list(df.columns)}")
    print("\n⚠️  SOLUTION:")
    print("   Rename your columns to: open, high, low, close, volume")
    exit(1)

print(f"✓ All required columns present")

# Check for timestamp
if 'timestamp' not in df.columns and 'date' in df.columns:
    df.rename(columns={'date': 'timestamp'}, inplace=True)

if 'timestamp' not in df.columns:
    print("⚠️  Warning: No timestamp column found")
    print("   Creating dummy timestamp...")
    df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='5min')

print()

# ============================================================
# TEST 5: Data Quality
# ============================================================
print("TEST 5: Checking data quality...")

# Check for NaNs
nan_counts = df[required_cols].isna().sum()
if nan_counts.sum() > 0:
    print(f"⚠️  Warning: Found NaN values:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"     {col}: {count} NaNs ({count/len(df)*100:.1f}%)")
else:
    print(f"✓ No NaN values in OHLCV data")

# Check for zeros
zero_counts = (df[required_cols] == 0).sum()
if zero_counts.sum() > 0:
    print(f"⚠️  Warning: Found zero values:")
    for col, count in zero_counts.items():
        if count > 0:
            print(f"     {col}: {count} zeros ({count/len(df)*100:.1f}%)")

# Check data range
print(f"\n✓ Data ranges:")
for col in required_cols:
    print(f"  {col:8} - Min: {df[col].min():12.2f}, Max: {df[col].max():12.2f}")

print()

# ============================================================
# TEST 6: Feature Engineering (Small Sample)
# ============================================================
print("TEST 6: Testing feature engineering on sample...")

# Take a sample to test
sample_size = min(5000, len(df))
df_sample = df.head(sample_size).copy()

print(f"  Using {sample_size} rows for testing...")

try:
    from feature_engineering import AdvancedFeatureEngineer

    feature_engineer = AdvancedFeatureEngineer(config)
    df_features = feature_engineer.engineer_all_features(df_sample)

    print(f"✓ Feature engineering successful")
    print(f"  Total features created: {len(df_features.columns)}")

    # Check for NaNs
    initial_len = len(df_features)
    nan_ratio = df_features.isna().sum().sum() / (len(df_features) * len(df_features.columns))
    print(f"  NaN ratio: {nan_ratio*100:.1f}%")

    # Clean data
    threshold = len(df_features.columns) * 0.5
    df_features = df_features.dropna(thresh=threshold)
    df_features = df_features.ffill().bfill()

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].median())
    df_features = df_features.dropna()

    final_len = len(df_features)
    print(f"  Rows after cleaning: {final_len:,} (dropped {initial_len - final_len:,}, {(initial_len-final_len)/initial_len*100:.1f}%)")

    if final_len == 0:
        print(f"❌ ERROR: All rows were dropped!")
        print(f"   This means features created too many NaNs")
        print("\n⚠️  SOLUTION:")
        print("   1. Use more data (at least 1000 bars)")
        print("   2. Reduce rolling window sizes in config")
        exit(1)

except Exception as e:
    print(f"❌ Error in feature engineering: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# ============================================================
# TEST 7: Labeling (Small Sample)
# ============================================================
print("TEST 7: Testing Triple Barrier labeling...")

try:
    from labeling import TripleBarrierLabeler

    labeler = TripleBarrierLabeler(config)
    df_labeled = labeler.create_labels(df_features, direction='both')

    long_signals = (df_labeled['target_long'] == 1).sum()
    short_signals = (df_labeled['target_short'] == 1).sum()

    print(f"✓ Labeling successful")
    print(f"  LONG signals: {long_signals} ({long_signals/len(df_labeled)*100:.1f}%)")
    print(f"  SHORT signals: {short_signals} ({short_signals/len(df_labeled)*100:.1f}%)")

    if long_signals == 0 and short_signals == 0:
        print(f"\n⚠️  WARNING: No signals generated!")
        print(f"   This means barriers are too tight")
        print("\n⚠️  SOLUTION:")
        print("   In config.py, try:")
        print("   - STATIC_TP_PCT = 1.0 (lower)")
        print("   - MIN_RETURN_THRESHOLD = 0.2 (lower)")

except Exception as e:
    print(f"❌ Error in labeling: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80 + "\n")

print("✅ All tests passed!")
print(f"\nYour data looks good:")
print(f"  - Total bars: {len(df):,}")
print(f"  - After feature engineering: ~{final_len:,} bars")
print(f"  - LONG signals: {long_signals} ({long_signals/len(df_labeled)*100:.1f}%)")
print(f"  - SHORT signals: {short_signals} ({short_signals/len(df_labeled)*100:.1f}%)")

print(f"\n🚀 Ready to train full model!")
print(f"   Run: python train_advanced.py")

print("\n" + "=" * 80)
