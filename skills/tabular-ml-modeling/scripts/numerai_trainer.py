#!/usr/bin/env python3
"""
Numerai Deep Model Trainer
Optimized for large tabular datasets with time-ordered validation.
Uses embargo CV to prevent temporal leakage.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
import gc
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    filename='numerai.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log(msg: str):
    """Log to file and print."""
    logging.info(msg)
    print(msg)

# Numerai scoring
try:
    from numerai_tools.scoring import correlation_contribution, numerai_corr
    NUMERAI_TOOLS_AVAILABLE = True
except ImportError:
    NUMERAI_TOOLS_AVAILABLE = False
    log("Warning: numerai-tools not installed. Install with: pip install numerai-tools")

# ML frameworks
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# =============================================================================
# TIME-ORDERED CV WITH EMBARGO
# =============================================================================

def make_embargo_splits(
    era: pd.Series, 
    n_splits: int = 5, 
    embargo_eras: int = 4,
    min_train_ratio: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Time-ordered CV with embargo between train and val.
    
    CRITICAL: GroupKFold causes severe temporal leakage!
    This function ensures training data always comes BEFORE validation data.
    
    Args:
        era: Series of era values
        n_splits: Number of CV folds
        embargo_eras: Number of eras to skip between train and val (recommend 4 for 20-day targets)
        min_train_ratio: Fraction of eras for initial training set
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    unique_eras = np.sort(era.unique())
    n_eras = len(unique_eras)
    
    min_train_eras = int(n_eras * min_train_ratio)
    remaining_eras = n_eras - min_train_eras - embargo_eras
    test_size = remaining_eras // n_splits
    
    era_to_idx = {e: np.where(era == e)[0] for e in unique_eras}
    
    splits = []
    for i in range(n_splits):
        train_end_idx = min_train_eras + (i * test_size)
        test_start_idx = train_end_idx + embargo_eras
        test_end_idx = test_start_idx + test_size
        
        if test_end_idx > n_eras:
            test_end_idx = n_eras
        if test_start_idx >= n_eras:
            break
            
        train_eras = unique_eras[:train_end_idx]
        test_eras = unique_eras[test_start_idx:test_end_idx]
        
        train_idx = np.concatenate([era_to_idx[e] for e in train_eras])
        test_idx = np.concatenate([era_to_idx[e] for e in test_eras])
        
        splits.append((np.sort(train_idx), np.sort(test_idx)))
        
        log(f"Fold {i+1}: Train eras {train_eras.min()}-{train_eras.max()}, "
            f"Val eras {test_eras.min()}-{test_eras.max()} (embargo={embargo_eras})")
    
    return splits


# =============================================================================
# DEEP MODEL CONFIGURATIONS
# =============================================================================

def get_lgb_params(seed: int = 42) -> dict:
    """LightGBM parameters - CPU only (GPU not supported on this platform)."""
    return {
        "objective": "regression",
        "metric": "mse",
        "device": "cpu",  # GPU not available
        "n_estimators": 2_000,  # Keep smaller for CPU
        "learning_rate": 0.01,
        "max_depth": 8,
        "num_leaves": 255,
        "min_data_in_leaf": 1_000,
        "feature_fraction": 0.1,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": seed,
    }


def get_xgb_deep_params(seed: int = 42) -> dict:
    """XGBoost deep parameters for Numerai-style data."""
    return {
        "objective": "reg:squarederror",
        "device": "cuda",
        "tree_method": "hist",
        "n_estimators": 30_000,
        "learning_rate": 0.001,
        "max_depth": 10,
        "max_leaves": 1024,
        "min_child_weight": 10_000,
        "colsample_bytree": 0.1,
        "subsample": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.1,
        "random_state": seed,
        "verbosity": 1,
    }


def get_cat_deep_params(seed: int = 42) -> dict:
    """CatBoost deep parameters for Numerai-style data."""
    return {
        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "iterations": 30_000,
        "learning_rate": 0.001,
        "depth": 10,
        "min_data_in_leaf": 10_000,
        "rsm": 0.1,  # colsample_bytree equivalent
        "subsample": 0.8,
        "l2_leaf_reg": 0.1,
        "random_seed": seed,
        "verbose": 1000,
    }


def get_lgb_light_params(seed: int = 42) -> dict:
    """Light parameters for quick exploration."""
    return {
        "objective": "regression",
        "metric": "mse",
        "device": "gpu",
        "n_estimators": 2_000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 15,
        "min_data_in_leaf": 1_000,
        "feature_fraction": 0.1,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": seed,
    }


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def rank_keep_ties_keep_na(s: pd.Series) -> pd.Series:
    """Rank with ties averaged, NaN preserved."""
    return s.rank(method='average', na_option='keep') / (s.notna().sum() + 1)


def rank_gauss_pow1(s: pd.Series) -> pd.Series:
    """Rank-normalize and gaussianize a series."""
    s_rank = rank_keep_ties_keep_na(s)
    s_rank_norm = pd.Series(stats.norm.ppf(s_rank), index=s_rank.index)
    result_series = s_rank_norm / s_rank_norm.std()
    return result_series


def evaluate_numerai(
    preds: np.ndarray, 
    target: np.ndarray, 
    benchmark: np.ndarray,
    era: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate predictions using Numerai payout formula.
    
    Payout = 0.75 * numerai_corr + 2.25 * correlation_contribution
    
    API signatures:
    - numerai_corr(predictions: DataFrame, targets: Series) -> Series
    - correlation_contribution(predictions: DataFrame, meta_model: Series, live_targets: Series) -> Series
    
    If era is provided, computes per-era scores then averages (correct method).
    If era is None, computes global score (less accurate but useful for quick checks).
    """
    if not NUMERAI_TOOLS_AVAILABLE:
        # Fallback to simple correlation
        corr = np.corrcoef(preds, target)[0, 1]
        return {
            'correlation': corr,
            'numerai_corr': np.nan,
            'correlation_contribution': np.nan,
            'payout': np.nan,
        }
    
    pred_col = 'prediction'
    
    if era is not None:
        # Correct method: per-era scores, then average
        df = pd.DataFrame({
            pred_col: preds,
            'target': target,
            'benchmark': benchmark,
            'era': era
        })
        
        # Note: numerai_corr expects DataFrame for predictions, Series for target
        nc_per_era = df.groupby('era').apply(
            lambda d: numerai_corr(d[[pred_col]], d['target'])
        )
        cc_per_era = df.groupby('era').apply(
            lambda d: correlation_contribution(d[[pred_col]], d['benchmark'], d['target'])
        )
        
        nc = nc_per_era[pred_col].mean()
        cc = cc_per_era[pred_col].mean()
    else:
        # Global score (less accurate, for quick checks only)
        preds_df = pd.DataFrame({pred_col: preds})
        target_s = pd.Series(target)
        benchmark_s = pd.Series(benchmark)
        
        nc = numerai_corr(preds_df, target_s)[pred_col]
        cc = correlation_contribution(preds_df, benchmark_s, target_s)[pred_col]
    
    payout = 0.75 * nc + 2.25 * cc
    
    return {
        'numerai_corr': float(nc),
        'correlation_contribution': float(cc),
        'payout': float(payout),
    }


def evaluate_per_era(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    benchmark_col: str,
    era_col: str = 'era'
) -> pd.DataFrame:
    """Evaluate predictions per era."""
    results = []
    
    for era in df[era_col].unique():
        era_mask = df[era_col] == era
        era_df = df[era_mask]
        
        scores = evaluate_numerai(
            era_df[pred_col].values,
            era_df[target_col].values,
            era_df[benchmark_col].values
        )
        scores['era'] = era
        results.append(scores)
    
    return pd.DataFrame(results)


# =============================================================================
# TRAINING CLASS
# =============================================================================

class NumeraiTrainer:
    """
    Deep model trainer for Numerai-style competitions.
    
    Features:
    - Era-based GroupKFold cross-validation
    - Deep model configurations
    - Multi-target training
    - Rank-gaussianization ensembling
    - Numerai payout scoring
    """
    
    TARGETS = [
        "target_ender_20",      # Main target
        "target_ender_60",
        "target_jasper_20",
        "target_jasper_60",
        "target_teager2b_20",
        "target_teager2b_60",
    ]
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_eras: int = 4,
        random_state: int = 42,
        use_deep_params: bool = False,  # Start light, progress to deep
        verbose: bool = True,
        save_checkpoints: bool = True,
    ):
        self.n_splits = n_splits
        self.embargo_eras = embargo_eras
        self.random_state = random_state
        self.use_deep_params = use_deep_params
        self.verbose = verbose
        self.save_checkpoints = save_checkpoints
        
        self.models = {}
        self.oof_predictions = {}
        self.cv_scores = {}
    
    def _get_splits(self, era: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get time-ordered CV splits with embargo."""
        return make_embargo_splits(
            era, 
            n_splits=self.n_splits, 
            embargo_eras=self.embargo_eras
        )
    
    def _log(self, msg: str):
        """Log to file and print if verbose."""
        log(msg)
    
    def _save_checkpoint(self, model, name: str, oof: np.ndarray, fold: int):
        """Save model checkpoint."""
        if self.save_checkpoints:
            joblib.dump(model, f'{name}_fold{fold}.pkl')
            np.save(f'{name}_oof_fold{fold}.npy', oof)
    
    def train_lgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        era: pd.Series,
        target_name: str = "target",
        benchmark: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List]:
        """Train LightGBM with embargo CV (CPU only)."""
        self._log(f"\n{'='*60}")
        self._log(f"Training LightGBM for {target_name} (CPU)")
        self._log(f"{'='*60}")
        
        # LightGBM is CPU only on this platform - keep light params
        params = get_lgb_params(self.random_state)
        self._log("Using CPU parameters (2K trees, LR=0.01)")
        
        splits = self._get_splits(era)
        oof = np.zeros(len(X))
        models = []
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            self._log(f"\nFold {fold + 1}/{len(splits)}")
            self._log(f"  Train: {len(train_idx):,} samples, Val: {len(val_idx):,} samples")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            
            callbacks = [
                lgb.early_stopping(500, verbose=False),
                lgb.log_evaluation(period=5000)
            ]
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            oof[val_idx] = model.predict(X_val)
            models.append(model)
            
            # Evaluate fold
            if benchmark is not None and NUMERAI_TOOLS_AVAILABLE:
                scores = evaluate_numerai(oof[val_idx], y_val, benchmark[val_idx])
                self._log(f"  Fold {fold+1} - NC: {scores['numerai_corr']:.4f}, "
                         f"CC: {scores['correlation_contribution']:.4f}, "
                         f"Payout: {scores['payout']:.4f}")
                fold_scores.append(scores)
            else:
                corr = np.corrcoef(oof[val_idx], y_val)[0, 1]
                self._log(f"  Fold {fold+1} Correlation: {corr:.4f}")
        
        # Overall CV score
        if benchmark is not None and NUMERAI_TOOLS_AVAILABLE:
            overall = evaluate_numerai(oof, y, benchmark)
            self._log(f"\nOverall CV - NC: {overall['numerai_corr']:.4f}, "
                     f"CC: {overall['correlation_contribution']:.4f}, "
                     f"Payout: {overall['payout']:.4f}")
        
        return oof, models
    
    def train_xgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        target_name: str = "target",
        benchmark: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List]:
        """Train XGBoost with era-based CV."""
        self._log(f"\n{'='*60}")
        self._log(f"Training XGBoost for {target_name}")
        self._log(f"{'='*60}")
        
        params = get_xgb_deep_params(self.random_state) if self.use_deep_params else {}
        
        cv = self._get_cv()
        oof = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
            self._log(f"\nFold {fold + 1}/{self.n_splits}")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=1000
            )
            
            oof[val_idx] = model.predict(X_val)
            models.append(model)
            
            if benchmark is not None and NUMERAI_TOOLS_AVAILABLE:
                scores = evaluate_numerai(oof[val_idx], y_val, benchmark[val_idx])
                self._log(f"  Fold {fold+1} Payout: {scores['payout']:.4f}")
        
        return oof, models
    
    def train_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        target_name: str = "target",
        benchmark: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List]:
        """Train CatBoost with era-based CV."""
        self._log(f"\n{'='*60}")
        self._log(f"Training CatBoost for {target_name}")
        self._log(f"{'='*60}")
        
        params = get_cat_deep_params(self.random_state) if self.use_deep_params else {}
        
        cv = self._get_cv()
        oof = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
            self._log(f"\nFold {fold + 1}/{self.n_splits}")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=500,
            )
            
            oof[val_idx] = model.predict(X_val)
            models.append(model)
        
        return oof, models
    
    def train_all_targets(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        benchmark_col: str = "v52_lgbm_ender20",
        era_col: str = "era",
        framework: str = "lgb",
    ) -> Dict[str, np.ndarray]:
        """
        Train models for all targets.
        
        Args:
            df: DataFrame with features, targets, era, and benchmark
            feature_cols: List of feature column names
            benchmark_col: Benchmark prediction column
            era_col: Era column for grouping
            framework: 'lgb', 'xgb', or 'cat'
        
        Returns:
            Dictionary of OOF predictions for each target
        """
        # Keep int8 - tree models handle it natively, saves 4x memory
        X = df[feature_cols].values
        groups = df[era_col].values
        benchmark = df[benchmark_col].values if benchmark_col in df.columns else None
        
        train_func = {
            'lgb': self.train_lgb,
            'xgb': self.train_xgb,
            'cat': self.train_catboost,
        }[framework]
        
        all_oof = {}
        all_models = {}
        
        for target in self.TARGETS:
            if target not in df.columns:
                self._log(f"Skipping {target} - not found in DataFrame")
                continue
            
            y = df[target].values
            oof, models = train_func(X, y, groups, target, benchmark)
            
            all_oof[target] = oof
            all_models[target] = models
            
            gc.collect()
        
        self.oof_predictions = all_oof
        self.models = all_models
        
        return all_oof
    
    def ensemble_predictions(
        self,
        df: pd.DataFrame,
        oof_preds: Dict[str, np.ndarray],
        weights: Dict[str, float],
        era_col: str = "era",
    ) -> np.ndarray:
        """
        Ensemble predictions using rank-gaussianization per era.
        
        Args:
            df: DataFrame with era column
            oof_preds: Dictionary of OOF predictions per target
            weights: Weight for each target
            era_col: Era column name
        
        Returns:
            Final ensembled predictions
        """
        # Build prediction DataFrame
        pred_df = pd.DataFrame(oof_preds)
        pred_df[era_col] = df[era_col].values
        
        # Rank-gaussianize each prediction per era
        pred_cols = list(oof_preds.keys())
        for col in pred_cols:
            pred_df[col] = pred_df.groupby(era_col, group_keys=False)[col].transform(rank_gauss_pow1)
        
        # Weighted combination
        weight_vec = [weights.get(col, 0.0) for col in pred_cols]
        weight_vec = np.array(weight_vec) / np.sum(weight_vec)  # Normalize
        
        ensemble = pred_df[pred_cols].values @ weight_vec
        
        # Final rank-gaussianization
        pred_df['ensemble'] = ensemble
        final = pred_df.groupby(era_col, group_keys=False)['ensemble'].transform(rank_gauss_pow1)
        
        return final.values


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Numerai Deep Model Trainer")
    print("="*60)
    
    # Example usage (replace with actual data loading)
    print("\nExample usage:")
    print("""
    # Load data
    df = pd.read_parquet("dataset.parquet")
    benchmark = pd.read_parquet("benchmark.parquet")
    df = df.merge(benchmark, on="id")
    
    # Get feature columns (exclude era, targets, id)
    # Keep int8 dtype - tree models handle it natively
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    
    # Initialize trainer
    trainer = NumeraiTrainer(
        n_splits=5,
        use_deep_params=True,  # Use 30K trees
        verbose=True
    )
    
    # Train all targets with LightGBM
    oof_preds = trainer.train_all_targets(
        df, 
        feature_cols,
        benchmark_col="v52_lgbm_ender20",
        framework="lgb"
    )
    
    # Ensemble with weights
    weights = {
        "target_ender_20": 0.7,
        "target_ender_60": 0.1,
        "target_jasper_20": 0.05,
        "target_jasper_60": 0.05,
        "target_teager2b_20": 0.05,
        "target_teager2b_60": 0.05,
    }
    
    final_preds = trainer.ensemble_predictions(df, oof_preds, weights)
    
    # Evaluate
    scores = evaluate_numerai(
        final_preds,
        df["target_ender_20"].values,
        df["v52_lgbm_ender20"].values
    )
    print(f"Final Payout: {scores['payout']:.4f}")
    """)
