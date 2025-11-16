"""
Advanced Model Training with Hyperparameter Optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Advanced model training with:
    - Hyperparameter optimization (Optuna)
    - Time series cross-validation
    - Walk-forward validation
    - Class imbalance handling
    - Probability calibration
    """

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.results = {}

    def prepare_data(self, df, feature_cols, target_col):
        """Prepare features and target"""
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Remove any remaining NaNs
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        # Replace inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        return X, y

    def train_model(self, X, y, model_name='xgboost', optimize=True):
        """
        Train a single model with optional hyperparameter optimization

        Args:
            X: Features
            y: Target
            model_name: 'xgboost', 'lightgbm', 'catboost', or 'random_forest'
            optimize: Whether to use Optuna for hyperparameter tuning

        Returns:
            Trained model
        """
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*80}\n")

        # Split data
        test_size = self.config.TEST_SIZE
        val_size = self.config.VALIDATION_SIZE

        n = len(X)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        print(f"Train samples: {len(X_train):,} (Class 1: {y_train.sum():,}, {y_train.mean()*100:.1f}%)")
        print(f"Val samples:   {len(X_val):,} (Class 1: {y_val.sum():,}, {y_val.mean()*100:.1f}%)")
        print(f"Test samples:  {len(X_test):,} (Class 1: {y_test.sum():,}, {y_test.mean()*100:.1f}%)\n")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Handle class imbalance
        if self.config.USE_SMOTE and y_train.sum() > 10:
            print("Applying SMOTE for class imbalance...")
            try:
                smote = SMOTE(random_state=self.config.RANDOM_STATE, k_neighbors=min(5, y_train.sum()-1))
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE: {len(y_train):,} samples (Class 1: {y_train.sum():,}, {y_train.mean()*100:.1f}%)\n")
            except Exception as e:
                print(f"SMOTE failed: {e}. Continuing without it.\n")

        # Get model
        if optimize and self.config.USE_OPTUNA:
            print("Optimizing hyperparameters with Optuna...\n")
            model = self._optimize_hyperparameters(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                model_name
            )
        else:
            model = self._get_default_model(model_name)
            print("Training with default hyperparameters...\n")

        # Train
        if model_name in ['xgboost', 'lightgbm', 'catboost']:
            # Use early stopping with validation set
            model = self._train_with_early_stopping(
                model, model_name,
                X_train_scaled, y_train,
                X_val_scaled, y_val
            )
        else:
            # Regular fit
            sample_weights = self._get_sample_weights(y_train) if self.config.USE_CLASS_WEIGHTS else None
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        # Calibrate probabilities
        if self.config.USE_CALIBRATION:
            print("\nCalibrating probabilities...")
            model = CalibratedClassifierCV(
                model,
                method=self.config.CALIBRATION_METHOD,
                cv='prefit'
            )
            model.fit(X_val_scaled, y_val)

        # Evaluate
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")

        results = {}
        for split_name, X_split, y_split in [
            ('Train', X_train_scaled, y_train),
            ('Val', X_val_scaled, y_val),
            ('Test', X_test_scaled, y_test)
        ]:
            metrics = self._evaluate_model(model, X_split, y_split)
            results[split_name] = metrics

            print(f"{split_name:6} | Acc: {metrics['accuracy']:.3f} | "
                  f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1']:.3f} | AUC: {metrics['roc_auc']:.3f} | "
                  f"MCC: {metrics['mcc']:.3f}")

        # Store
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.results[model_name] = results

        return model, scaler, results

    def _get_default_model(self, model_name):
        """Get model with default hyperparameters"""
        if model_name == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                eval_metric='logloss',
                verbosity=0
            )
        elif model_name == 'lightgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                verbosity=-1
            )
        elif model_name == 'catboost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                random_state=self.config.RANDOM_STATE,
                verbose=0,
                thread_count=self.config.N_JOBS
            )
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _optimize_hyperparameters(self, X_train, y_train, X_val, y_val, model_name):
        """Optimize hyperparameters using Optuna"""
        import optuna
        from optuna.samplers import TPESampler

        def objective(trial):
            # Suggest hyperparameters
            if model_name == 'xgboost':
                from xgboost import XGBClassifier
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': self.config.RANDOM_STATE,
                    'n_jobs': self.config.N_JOBS,
                    'eval_metric': 'logloss',
                    'verbosity': 0
                }
                model = XGBClassifier(**params)

            elif model_name == 'lightgbm':
                from lightgbm import LGBMClassifier
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.config.RANDOM_STATE,
                    'n_jobs': self.config.N_JOBS,
                    'verbosity': -1
                }
                model = LGBMClassifier(**params)

            elif model_name == 'catboost':
                from catboost import CatBoostClassifier
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 10),
                    'random_state': self.config.RANDOM_STATE,
                    'verbose': 0,
                    'thread_count': self.config.N_JOBS
                }
                model = CatBoostClassifier(**params)

            # Train and evaluate
            sample_weights = self._get_sample_weights(y_train) if self.config.USE_CLASS_WEIGHTS else None
            model.fit(X_train, y_train, sample_weight=sample_weights)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)

            return score

        # Run optimization
        sampler = TPESampler(seed=self.config.RANDOM_STATE)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        study.optimize(
            objective,
            n_trials=self.config.OPTUNA_TRIALS,
            show_progress_bar=True,
            n_jobs=1  # Optuna will parallelize internally if needed
        )

        print(f"\nBest AUC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}\n")

        # Train final model with best params
        best_params = study.best_params
        best_params.update({
            'random_state': self.config.RANDOM_STATE,
            'n_jobs': self.config.N_JOBS
        })

        if model_name == 'xgboost':
            from xgboost import XGBClassifier
            best_params['eval_metric'] = 'logloss'
            best_params['verbosity'] = 0
            model = XGBClassifier(**best_params)
        elif model_name == 'lightgbm':
            from lightgbm import LGBMClassifier
            best_params['verbosity'] = -1
            model = LGBMClassifier(**best_params)
        elif model_name == 'catboost':
            from catboost import CatBoostClassifier
            best_params['verbose'] = 0
            best_params['thread_count'] = self.config.N_JOBS
            model = CatBoostClassifier(**best_params)

        return model

    def _train_with_early_stopping(self, model, model_name, X_train, y_train, X_val, y_val):
        """Train gradient boosting models with early stopping"""
        sample_weights = self._get_sample_weights(y_train) if self.config.USE_CLASS_WEIGHTS else None

        if model_name == 'xgboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                verbose=False
            )
        elif model_name == 'lightgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                callbacks=[
                    __import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False),
                    __import__('lightgbm').log_evaluation(period=0)
                ]
            )
        elif model_name == 'catboost':
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                sample_weight=sample_weights,
                verbose=False,
                early_stopping_rounds=50
            )

        return model

    def _get_sample_weights(self, y):
        """Calculate class weights"""
        from sklearn.utils.class_weight import compute_sample_weight
        return compute_sample_weight('balanced', y)

    def _evaluate_model(self, model, X, y):
        """Evaluate model on given dataset"""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5,
            'average_precision': average_precision_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0,
            'mcc': matthews_corrcoef(y, y_pred)
        }

        return metrics

    def time_series_cross_validation(self, X, y, model_name='xgboost'):
        """
        Perform time series cross-validation
        """
        print(f"\n{'='*80}")
        print(f"Time Series Cross-Validation - {model_name.upper()}")
        print(f"{'='*80}\n")

        tscv = TimeSeriesSplit(
            n_splits=self.config.N_SPLITS,
            gap=self.config.GAP_SIZE
        )

        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}/{self.config.N_SPLITS}")
            print(f"  Train: {len(train_idx):,} samples, Test: {len(test_idx):,} samples")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            model = self._get_default_model(model_name)
            sample_weights = self._get_sample_weights(y_train) if self.config.USE_CLASS_WEIGHTS else None
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

            # Evaluate
            metrics = self._evaluate_model(model, X_test_scaled, y_test)
            metrics['fold'] = fold
            cv_results.append(metrics)

            print(f"  Test F1: {metrics['f1']:.3f}, AUC: {metrics['roc_auc']:.3f}")

        # Summary
        cv_df = pd.DataFrame(cv_results)
        print(f"\n{'='*80}")
        print("Cross-Validation Summary")
        print(f"{'='*80}\n")
        print(cv_df.describe().loc[['mean', 'std']])

        return cv_df

    def walk_forward_validation(self, X, y, model_name='xgboost'):
        """
        Perform walk-forward validation (more realistic for trading)
        """
        print(f"\n{'='*80}")
        print(f"Walk-Forward Validation - {model_name.upper()}")
        print(f"{'='*80}\n")

        window_size = self.config.WALK_FORWARD_WINDOW
        step_size = self.config.WALK_FORWARD_STEP

        results = []
        predictions = []

        n = len(X)
        start = 0

        fold = 1
        while start + window_size < n:
            train_end = start + window_size
            test_end = min(train_end + step_size, n)

            print(f"\nFold {fold}")
            print(f"  Train: [{start}:{train_end}] ({train_end - start:,} samples)")
            print(f"  Test:  [{train_end}:{test_end}] ({test_end - train_end:,} samples)")

            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            model = self._get_default_model(model_name)
            sample_weights = self._get_sample_weights(y_train) if self.config.USE_CLASS_WEIGHTS else None
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

            # Predict
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Store predictions
            for i, idx in enumerate(range(train_end, test_end)):
                predictions.append({
                    'index': idx,
                    'y_true': y_test.iloc[i],
                    'y_pred_proba': y_pred_proba[i],
                    'fold': fold
                })

            # Evaluate
            metrics = self._evaluate_model(model, X_test_scaled, y_test)
            metrics['fold'] = fold
            metrics['train_start'] = start
            metrics['train_end'] = train_end
            metrics['test_end'] = test_end
            results.append(metrics)

            print(f"  Test F1: {metrics['f1']:.3f}, AUC: {metrics['roc_auc']:.3f}")

            # Move forward
            start += step_size
            fold += 1

        # Summary
        results_df = pd.DataFrame(results)
        predictions_df = pd.DataFrame(predictions)

        print(f"\n{'='*80}")
        print("Walk-Forward Validation Summary")
        print(f"{'='*80}\n")
        print(results_df.describe().loc[['mean', 'std']])

        return results_df, predictions_df
