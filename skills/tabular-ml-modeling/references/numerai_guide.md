# Numerai Competition Guide

## Dataset Structure

```
dataset.parquet (4.5M rows):
├── id: unique row identifier
├── era: date indicator (GROUP KEY for CV, NOT a feature)
├── feature_*: 2748 features, int8, values in [0,1,2,3,4]
├── target_ender_20: MAIN TARGET (float32)
└── auxiliary targets: target_ender_60, target_jasper_20, 
    target_jasper_60, target_teager2b_20, target_teager2b_60

benchmark.parquet (4.5M rows):
├── id: matches dataset.parquet
└── v52_lgbm_ender20: benchmark predictions for correlation_contribution
```

## Critical Rules

1. **Never use `era` as a feature** - it's only for grouping
2. **Use GroupKFold by era** - prevents temporal leakage
3. **Features are pre-quantized** - int8 [0-4], no scaling needed
4. **Deep models win** - this dataset rewards depth over breadth

## Payout Formula

```python
from numerai_tools.scoring import numerai_corr, correlation_contribution

# API signatures:
# numerai_corr(predictions: DataFrame, targets: Series) -> Series
# correlation_contribution(predictions: DataFrame, meta_model: Series, live_targets: Series) -> Series

pred_col = 'prediction'

# Compute per-era, then average
per_era_nc = df.groupby('era').apply(
    lambda d: numerai_corr(d[[pred_col]], d['target_ender_20'])
)
per_era_cc = df.groupby('era').apply(
    lambda d: correlation_contribution(d[[pred_col]], d['v52_lgbm_ender20'], d['target_ender_20'])
)

nc = per_era_nc[pred_col].mean()
cc = per_era_cc[pred_col].mean()
payout = 0.75 * nc + 2.25 * cc
```

**Key insight**: correlation_contribution is weighted 3x more than raw correlation!
- This rewards predictions that add value BEYOND the benchmark
- Optimize for orthogonal signal, not just correlation

### Target Metrics
| Metric | Acceptable | Extraordinary |
|--------|------------|---------------|
| numerai_corr | 0.030 | 0.055 |
| correlation_contribution | 0.010 | 0.035 |

## Deep Model Parameters

This dataset requires unusually deep models. **But start light and progress gradually.**

### Progressive Training Strategy
1. Start with n_estimators=2000, learning_rate=0.01
2. Use early stopping to find optimal tree count
3. Gradually reduce LR and increase trees
4. Target: 30K trees with LR=0.001 for GPU models

### GPU Availability
| Framework | GPU | Recommendation |
|-----------|-----|----------------|
| XGBoost | ✅ CUDA | Go deep (30K trees) |
| CatBoost | ✅ GPU | Go deep (30K trees) |
| LightGBM | ❌ CPU only | Keep light (2K trees) |

### XGBoost Deep Config (GPU)
```python
xgb_deep_params = {
    "device": "cuda",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_estimators": 30_000,
    "learning_rate": 0.001,
    "max_depth": 10,
    "max_leaves": 1024,
    "min_child_weight": 10_000,
    "colsample_bytree": 0.1,
    "subsample": 0.8,
    "reg_lambda": 0.1,
    "early_stopping_rounds": 500,
    "random_state": 42,
}
```

### CatBoost Deep Config (GPU)
```python
cat_deep_params = {
    "task_type": "GPU",
    "devices": "0",
    "loss_function": "RMSE",
    "iterations": 30_000,
    "learning_rate": 0.001,
    "depth": 10,
    "min_data_in_leaf": 10_000,
    "rsm": 0.1,
    "subsample": 0.8,
    "l2_leaf_reg": 0.1,
    "early_stopping_rounds": 500,
    "random_seed": 42,
    "verbose": 1000,
}
```

### LightGBM Config (CPU - Keep Light)
```python
lgb_params = {
    "device": "cpu",  # GPU not available on this platform
    "objective": "regression",
    "metric": "mse",
    "n_estimators": 2_000,  # Keep smaller for CPU
    "learning_rate": 0.01,
    "max_depth": 8,
    "num_leaves": 255,
    "min_data_in_leaf": 1_000,
    "feature_fraction": 0.1,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
```

### Light Parameters (Quick Exploration)
```python
light_params = {
    "n_estimators": 2_000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 15,
    "feature_fraction": 0.1,
    "min_data_in_leaf": 1_000,
    "early_stopping_rounds": 100,
}
```

## Validation Strategy

**CRITICAL: GroupKFold causes severe temporal leakage!**

Adjacent eras have ~99.5% target correlation (20-day forward returns with weekly eras = ~75% overlap). GroupKFold mixes future and past eras, causing inflated OOF scores (~10x higher than reality).

**Use time-ordered CV with embargo gap:**

```python
def make_embargo_splits(
    era: pd.Series, 
    n_splits: int = 5, 
    embargo_eras: int = 4,
    min_train_ratio: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Time-ordered CV with embargo between train and val.
    
    - Training data always comes BEFORE validation data temporally
    - Embargo gap of N eras between train and val prevents target leakage
    - Expanding window: training set grows with each fold
    
    Args:
        era: Series of era values
        n_splits: Number of CV folds
        embargo_eras: Number of eras to skip (recommend 4 for 20-day targets)
        min_train_ratio: Fraction of eras for initial training set
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
    
    return splits

# Usage - ALWAYS verify train comes before val
splits = make_embargo_splits(df['era'], n_splits=5, embargo_eras=4)
for i, (train_idx, val_idx) in enumerate(splits):
    train_eras = df.iloc[train_idx]['era']
    val_eras = df.iloc[val_idx]['era']
    assert train_eras.max() < val_eras.min(), f"Fold {i}: Train must come before val!"
```

### Expected Realistic Scores

With proper embargo CV, expect:
- numerai_corr: 0.03-0.05 (not 0.15+!)
- correlation_contribution: 0.01-0.02

If your OOF scores are 10x higher, you have leakage.

## Multi-Target Training Strategy

Train separate models on each target, then ensemble:

```python
targets = [
    "target_ender_20",      # MAIN - highest weight
    "target_ender_60",      # Auxiliary
    "target_jasper_20",     # Auxiliary  
    "target_jasper_60",     # Auxiliary
    "target_teager2b_20",   # Auxiliary
    "target_teager2b_60",   # Auxiliary
]

# Train model for each target
models = {}
oof_preds = {}
for target in targets:
    model, oof = train_with_cv(X, df[target], groups=df['era'])
    models[target] = model
    oof_preds[target] = oof
```

## Ensembling Protocol

**Critical**: Use rank-gaussianization per era before combining:

```python
from scipy import stats
import pandas as pd
import numpy as np

def rank_keep_ties_keep_na(s: pd.Series) -> pd.Series:
    """Rank with ties averaged, NaN preserved."""
    return s.rank(method='average', na_option='keep') / (s.notna().sum() + 1)

def rank_gauss_pow1(s: pd.Series) -> pd.Series:
    """Rank-normalize and gaussianize a series."""
    s_rank = rank_keep_ties_keep_na(s)
    s_rank_norm = pd.Series(stats.norm.ppf(s_rank), index=s_rank.index)
    # Standardize to std=1
    result_series = s_rank_norm / s_rank_norm.std()
    return result_series

def ensemble_predictions(df: pd.DataFrame, 
                         pred_cols: list, 
                         weights: list,
                         era_col: str = "era") -> pd.Series:
    """
    Ensemble predictions using rank-gaussianization per era.
    
    Args:
        df: DataFrame with predictions and era column
        pred_cols: List of prediction column names
        weights: Weight for each prediction (should sum to 1)
        era_col: Name of era column
    
    Returns:
        Final ensembled predictions (rank-gaussianized)
    """
    df = df.copy()
    
    # Step 1: Rank-gaussianize each prediction per era
    for col in pred_cols:
        df[col] = df.groupby(era_col, group_keys=False)[col].transform(rank_gauss_pow1)
    
    # Step 2: Weighted combination
    ensemble = df[pred_cols].dot(weights)
    
    # Step 3: Final rank-gaussianization per era
    df['ensemble'] = ensemble
    final = df.groupby(era_col, group_keys=False)['ensemble'].transform(rank_gauss_pow1)
    
    return final
```

### Weight Selection Strategy

The main target (target_ender_20) should have highest weight since payout directly measures it:

```python
# Conservative: Heavy main target
weights_conservative = {
    "target_ender_20": 0.7,
    "target_ender_60": 0.1,
    "target_jasper_20": 0.05,
    "target_jasper_60": 0.05,
    "target_teager2b_20": 0.05,
    "target_teager2b_60": 0.05,
}

# Exploratory: More diverse
weights_diverse = {
    "target_ender_20": 0.4,
    "target_ender_60": 0.2,
    "target_jasper_20": 0.1,
    "target_jasper_60": 0.1,
    "target_teager2b_20": 0.1,
    "target_teager2b_60": 0.1,
}
```

## Objective Function Selection

Since payout emphasizes correlation_contribution:

1. **MSE/RMSE** works as proxy for correlation (most common)
2. **Custom correlation loss** if framework supports it
3. **Consider Spearman correlation** for rank-based evaluation

```python
# For evaluation, use actual payout components
from numerai_tools.scoring import correlation_contribution, numerai_corr

def evaluate_payout(df, pred_col='prediction', target_col='target_ender_20', 
                    benchmark_col='v52_lgbm_ender20', era_col='era'):
    """
    Calculate expected payout score using per-era averaging.
    
    API signatures:
    - numerai_corr(predictions: DataFrame, targets: Series) -> Series
    - correlation_contribution(predictions: DataFrame, meta_model: Series, live_targets: Series) -> Series
    """
    per_era_nc = df.groupby(era_col).apply(
        lambda d: numerai_corr(d[[pred_col]], d[target_col])
    )
    per_era_cc = df.groupby(era_col).apply(
        lambda d: correlation_contribution(d[[pred_col]], d[benchmark_col], d[target_col])
    )
    
    nc = per_era_nc[pred_col].mean()
    cc = per_era_cc[pred_col].mean()
    payout = 0.75 * nc + 2.25 * cc
    
    return {
        'numerai_corr': float(nc),
        'correlation_contribution': float(cc),
        'payout': float(payout)
    }
```

## Memory Optimization for 4.5M Rows

Features are already int8, which is memory-efficient:
- 4.5M × 2748 × 1 byte ≈ 12.4 GB for features
- Fits comfortably in 80GB GPU and 160GB RAM

```python
# Load efficiently
import pandas as pd

# Features are int8 - keep them that way! 
# Tree models handle int8 natively, no conversion needed
df = pd.read_parquet("dataset.parquet")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

# Verify dtypes
assert df[feature_cols].dtypes.unique()[0] == 'int8'

# Use directly - do NOT convert to float32 (wastes 4x memory)
X = df[feature_cols].values  # Keep as int8
```

## Training Pipeline Summary

```python
# 1. Load data
df = pd.read_parquet("dataset.parquet")
benchmark = pd.read_parquet("benchmark.parquet")
df = df.merge(benchmark, on="id")

# 2. Prepare features (NO era, NO dtype conversion!)
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].values  # Keep int8 - tree models handle it natively

# 3. Train on each target with era-grouped CV
targets = ["target_ender_20", "target_ender_60", ...]
all_oof = {}

for target in targets:
    y = df[target].values
    oof, models = train_deep_model(X, y, groups=df['era'])
    all_oof[target] = oof

# 4. Ensemble with rank-gaussianization
pred_df = pd.DataFrame(all_oof)
pred_df['era'] = df['era']
weights = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]
final_preds = ensemble_predictions(pred_df, list(all_oof.keys()), weights)

# 5. Evaluate
scores = evaluate_payout(
    final_preds, 
    df['target_ender_20'], 
    df['v52_lgbm_ender20']
)
print(f"Payout: {scores['payout']:.4f}")
```
