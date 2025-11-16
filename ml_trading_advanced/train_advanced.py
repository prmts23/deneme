"""
Advanced ML Trading Strategy - Main Training Script

Usage:
    python train_advanced.py

This will:
1. Load and prepare data
2. Engineer advanced features
3. Create labels using Triple Barrier Method
4. Train models with hyperparameter optimization
5. Perform walk-forward validation
6. Save models and results
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Import our modules
import config
from feature_engineering import AdvancedFeatureEngineer
from labeling import TripleBarrierLabeler
from model_training import ModelTrainer


def main():
    print("=" * 140)
    print("ADVANCED ML TRADING STRATEGY - TRAINING PIPELINE")
    print("=" * 140)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("=" * 140)
    print("STEP 1: LOADING DATA")
    print("=" * 140 + "\n")

    # Update this path to your data
    data_path = config.DATA_PATH

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("\n⚠️  Please update DATA_PATH in config.py to point to your CSV file.")
        return

    df = pd.read_csv(data_path)

    # Standardize column names
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns and 'Timestamp' in df.columns:
        df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

    if 'timestamp' not in df.columns:
        print("❌ 'timestamp' column not found in data")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✓ Data loaded: {len(df):,} bars")
    print(f"  Symbol: {config.SYMBOL}")
    print(f"  Timeframe: {config.TIMEFRAME}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    print("=" * 140)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 140 + "\n")

    feature_engineer = AdvancedFeatureEngineer(config)
    df = feature_engineer.engineer_all_features(df)

    # Handle NaN values smartly (from rolling calculations)
    initial_len = len(df)

    # Strategy 1: Drop rows where >50% of columns are NaN
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)

    # Strategy 2: Forward fill remaining NaNs (for rolling features)
    df = df.ffill()

    # Strategy 3: Backward fill any remaining NaNs at the start
    df = df.bfill()

    # Strategy 4: Fill any remaining NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Final safety check - drop rows with any remaining NaNs
    df = df.dropna()

    print(f"\n✓ Features engineered. Rows after cleaning: {len(df):,} (dropped {initial_len - len(df):,})\n")

    if len(df) == 0:
        print("❌ ERROR: All data was dropped! This usually means:")
        print("   1. Not enough data (need at least 200 bars)")
        print("   2. Data quality issues (check for NaN/inf values)")
        print("   3. Rolling window periods too large")
        return

    # ========================================================================
    # 3. CREATE LABELS
    # ========================================================================
    print("=" * 140)
    print("STEP 3: CREATING LABELS (TRIPLE BARRIER METHOD)")
    print("=" * 140 + "\n")

    print(f"Barrier configuration:")
    print(f"  Method: {config.BARRIER_METHOD}")
    print(f"  Vertical barrier: {config.VERTICAL_BARRIER_HOURS} bars")
    print(f"  Dynamic barriers: {config.USE_DYNAMIC_BARRIERS}")
    if config.USE_DYNAMIC_BARRIERS:
        print(f"  Volatility lookback: {config.VOLATILITY_LOOKBACK} bars")
    else:
        print(f"  Static TP: {config.STATIC_TP_PCT}%")
        print(f"  Static SL: {config.STATIC_SL_PCT}%")
    print(f"  Min return threshold: {config.MIN_RETURN_THRESHOLD}%")
    print()

    labeler = TripleBarrierLabeler(config)
    df = labeler.create_labels(df, direction='both')

    # ========================================================================
    # 4. PREPARE FEATURES
    # ========================================================================
    print("=" * 140)
    print("STEP 4: PREPARING FEATURES")
    print("=" * 140 + "\n")

    # Get feature columns (exclude OHLCV, timestamp, targets, etc.)
    exclude_cols = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'target_long', 'target_short', 'return_long', 'return_short',
        'hour', 'dayofweek', 'day', 'volatility', 'sample_weight'
    ]

    all_features = [col for col in df.columns if col not in exclude_cols]

    print(f"Total features: {len(all_features)}")

    # Feature selection (optional - can use all features with tree models)
    if config.FEATURE_SELECTION_METHOD and len(all_features) > config.TOP_N_FEATURES:
        print(f"\nPerforming feature selection to get TOP {config.TOP_N_FEATURES} features...")

        # Quick feature selection using RandomForest importance
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        X_temp = df[all_features].copy()
        y_temp = df['target_long'].copy()

        # Remove any inf/nan
        X_temp = X_temp.replace([np.inf, -np.inf], np.nan)
        X_temp = X_temp.fillna(X_temp.median())

        scaler = StandardScaler()
        X_temp_scaled = scaler.fit_transform(X_temp)

        rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_temp_scaled, y_temp)

        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features = importance_df.head(config.TOP_N_FEATURES)['feature'].tolist()

        print(f"\n✓ Selected {len(selected_features)} features")
        print("\nTop 20 features:")
        for i, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']:50} {row['importance']:.6f}")

        # Save feature importance
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        importance_df.to_csv(f"{config.OUTPUT_DIR}/feature_importance_initial.csv", index=False)
    else:
        selected_features = all_features
        print(f"\nUsing all {len(selected_features)} features")

    print()

    # ========================================================================
    # 5. TRAIN MODELS
    # ========================================================================
    print("=" * 140)
    print("STEP 5: TRAINING MODELS")
    print("=" * 140 + "\n")

    trainer = ModelTrainer(config)

    # Train for LONG signals
    if 'target_long' in df.columns:
        print("\n" + "=" * 140)
        print("TRAINING LONG MODELS")
        print("=" * 140 + "\n")

        X_long, y_long = trainer.prepare_data(df, selected_features, 'target_long')

        long_models = {}
        long_scalers = {}

        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            if config.MODELS.get(model_name, False):
                try:
                    model, scaler, results = trainer.train_model(
                        X_long, y_long,
                        model_name=model_name,
                        optimize=config.USE_OPTUNA
                    )
                    long_models[model_name] = model
                    long_scalers[model_name] = scaler
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}\n")

        # Find best LONG model
        if trainer.results:
            best_long_model = max(
                trainer.results.items(),
                key=lambda x: x[1]['Test']['f1']
            )
            print(f"\n🏆 Best LONG model: {best_long_model[0].upper()}")
            print(f"   Test F1: {best_long_model[1]['Test']['f1']:.3f}")
            print(f"   Test AUC: {best_long_model[1]['Test']['roc_auc']:.3f}\n")

    # Train for SHORT signals
    if 'target_short' in df.columns:
        print("\n" + "=" * 140)
        print("TRAINING SHORT MODELS")
        print("=" * 140 + "\n")

        X_short, y_short = trainer.prepare_data(df, selected_features, 'target_short')

        # Reset trainer results for short models
        trainer.results = {}

        short_models = {}
        short_scalers = {}

        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            if config.MODELS.get(model_name, False):
                try:
                    model, scaler, results = trainer.train_model(
                        X_short, y_short,
                        model_name=model_name,
                        optimize=config.USE_OPTUNA
                    )
                    short_models[model_name] = model
                    short_scalers[model_name] = scaler
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}\n")

        # Find best SHORT model
        if trainer.results:
            best_short_model = max(
                trainer.results.items(),
                key=lambda x: x[1]['Test']['f1']
            )
            print(f"\n🏆 Best SHORT model: {best_short_model[0].upper()}")
            print(f"   Test F1: {best_short_model[1]['Test']['f1']:.3f}")
            print(f"   Test AUC: {best_short_model[1]['Test']['roc_auc']:.3f}\n")

    # ========================================================================
    # 6. WALK-FORWARD VALIDATION (Optional but recommended)
    # ========================================================================
    if config.USE_WALK_FORWARD:
        print("\n" + "=" * 140)
        print("STEP 6: WALK-FORWARD VALIDATION")
        print("=" * 140 + "\n")

        if 'target_long' in df.columns and X_long is not None:
            print("\n--- LONG MODEL ---")
            wf_results_long, wf_predictions_long = trainer.walk_forward_validation(
                X_long, y_long,
                model_name=best_long_model[0]
            )
            wf_results_long.to_csv(f"{config.OUTPUT_DIR}/walk_forward_long.csv", index=False)
            wf_predictions_long.to_csv(f"{config.OUTPUT_DIR}/walk_forward_predictions_long.csv", index=False)

        if 'target_short' in df.columns and X_short is not None:
            print("\n--- SHORT MODEL ---")
            wf_results_short, wf_predictions_short = trainer.walk_forward_validation(
                X_short, y_short,
                model_name=best_short_model[0]
            )
            wf_results_short.to_csv(f"{config.OUTPUT_DIR}/walk_forward_short.csv", index=False)
            wf_predictions_short.to_csv(f"{config.OUTPUT_DIR}/walk_forward_predictions_short.csv", index=False)

    # ========================================================================
    # 7. SAVE MODELS & ARTIFACTS
    # ========================================================================
    print("\n" + "=" * 140)
    print("STEP 7: SAVING MODELS & ARTIFACTS")
    print("=" * 140 + "\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save LONG models
    if long_models:
        for model_name, model in long_models.items():
            pickle.dump(model, open(f"{config.OUTPUT_DIR}/model_long_{model_name}.pkl", 'wb'))
            pickle.dump(long_scalers[model_name], open(f"{config.OUTPUT_DIR}/scaler_long_{model_name}.pkl", 'wb'))
            print(f"✓ Saved LONG {model_name} model and scaler")

    # Save SHORT models
    if short_models:
        for model_name, model in short_models.items():
            pickle.dump(model, open(f"{config.OUTPUT_DIR}/model_short_{model_name}.pkl", 'wb'))
            pickle.dump(short_scalers[model_name], open(f"{config.OUTPUT_DIR}/scaler_short_{model_name}.pkl", 'wb'))
            print(f"✓ Saved SHORT {model_name} model and scaler")

    # Save feature list
    with open(f"{config.OUTPUT_DIR}/features.txt", 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"✓ Saved feature list ({len(selected_features)} features)")

    # Save config
    with open(f"{config.OUTPUT_DIR}/config.txt", 'w') as f:
        for key, value in vars(config).items():
            if not key.startswith('_'):
                f.write(f"{key} = {value}\n")
    print("✓ Saved configuration")

    # ========================================================================
    # 8. FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 140)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 140 + "\n")

    print(f"Total bars processed: {len(df):,}")
    print(f"Features engineered: {len(selected_features)}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print()

    if 'target_long' in df.columns:
        print(f"LONG signals:")
        print(f"  Total: {(df['target_long'] == 1).sum():,} ({(df['target_long'] == 1).sum() / len(df) * 100:.2f}%)")
        if long_models:
            print(f"  Best model: {best_long_model[0]}")
            print(f"  Test F1: {best_long_model[1]['Test']['f1']:.3f}")
            print(f"  Test Precision: {best_long_model[1]['Test']['precision']:.3f}")
            print(f"  Test Recall: {best_long_model[1]['Test']['recall']:.3f}")
        print()

    if 'target_short' in df.columns:
        print(f"SHORT signals:")
        print(f"  Total: {(df['target_short'] == 1).sum():,} ({(df['target_short'] == 1).sum() / len(df) * 100:.2f}%)")
        if short_models:
            print(f"  Best model: {best_short_model[0]}")
            print(f"  Test F1: {best_short_model[1]['Test']['f1']:.3f}")
            print(f"  Test Precision: {best_short_model[1]['Test']['precision']:.3f}")
            print(f"  Test Recall: {best_short_model[1]['Test']['recall']:.3f}")
        print()

    print("=" * 140)
    print("✅ ALL DONE! Models are ready for deployment.")
    print("=" * 140)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("\n📝 NEXT STEPS:")
    print("  1. Review walk-forward validation results")
    print("  2. Adjust probability thresholds for your risk tolerance")
    print("  3. Implement in your trading system (Freqtrade, MT5, etc.)")
    print("  4. Start with paper trading!")
    print()


if __name__ == '__main__':
    main()
