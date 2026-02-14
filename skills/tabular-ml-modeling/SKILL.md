---
name: tabular-ml-modeling
description: State-of-the-art machine learning for large-scale tabular data (millions of rows, thousands of features). Use when building predictive models on tabular/structured data including regression and classification tasks. Covers GPU-accelerated gradient boosting (XGBoost, LightGBM, CatBoost), RAPIDS acceleration (cuDF, cuML), validation strategies, feature engineering, ensembling (hill climbing, stacking), and competition-winning techniques. Includes Numerai-specific optimizations with deep model parameters, era-based validation, multi-target ensembling, and payout scoring. Optimized for NVIDIA A100 GPU with CUDA 12.4.
---

# Tabular ML Modeling

Battle-tested techniques for large-scale tabular data modeling, refined from Kaggle Grandmaster competition experience and optimized for GPU acceleration.

## Environment Specifications

- **GPU**: NVIDIA A100-SXM4-80GB (80GB VRAM), CUDA 12.4
- **CPU**: 8 cores x64, 160GB RAM
- **Disk**: 200GB free
- **Dataset**: 4.5M rows, 2748 int8 features [0-4], era-based structure

## Platform Persistence (CRITICAL)

**The platform is ephemeral.** It shuts down when idle and all work is lost.

### Required Practices
1. **Save models frequently** to disk after each training run
2. **Maintain a log file** (`numerai.log`) with progress, findings, decisions
3. **Checkpoint OOF predictions** after each model completes
4. **Save ensemble weights** when optimizing

```python
import logging
import joblib

# Setup logging
logging.basicConfig(
    filename='numerai.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log(msg):
    logging.info(msg)
    print(msg)

# Save model after training
def save_checkpoint(model, name, oof_preds, scores):
    joblib.dump(model, f'{name}_model.pkl')
    np.save(f'{name}_oof.npy', oof_preds)
    log(f"Saved {name}: NC={scores['nc']:.4f}, CC={scores['cc']:.4f}")
```

## Core Principles

### Fast Experimentation
Maximize high-quality experiments. GPU acceleration transforms day-long iterations into minutes:
- Use RAPIDS cuDF for dataframe operations (up to 150x faster than pandas)
- Use GPU backends for XGBoost, LightGBM, CatBoost
- Use cuML for sklearn-compatible ML algorithms

### Robust Validation
Never trust a single train/test split. Use k-fold cross-validation matched to data structure:
- **Standard**: StratifiedKFold for classification, KFold for regression
- **Time-series**: TimeSeriesSplit
- **Grouped data**: GroupKFold (e.g., era-based predictions)

### Ensemble Everything
Single models leave performance on the table. Combine aggressively:
- **Diverse frameworks**: CatBoost + XGBoost + LightGBM capture different patterns
- **Multiple targets**: Auxiliary targets provide orthogonal signal
- **Hill climbing**: Optimize weights on OOF predictions
- **Stacking**: Meta-models can learn non-linear combinations

## Smarter EDA

Go beyond basic statistics. These checks catch problems that sink models:

### Train vs Test Distribution
```python
# Check for distribution shift that breaks generalization
import matplotlib.pyplot as plt

for col in important_features[:10]:
    fig, ax = plt.subplots()
    train[col].hist(ax=ax, alpha=0.5, label='train', bins=50)
    test[col].hist(ax=ax, alpha=0.5, label='test', bins=50)
    ax.legend()
    ax.set_title(f'{col} distribution')
```

### Temporal Patterns in Target
```python
# Check for trends/seasonality that require time-aware validation
df.groupby('era')['target_ender_20'].mean().plot()
plt.title('Target mean by era - look for trends')
```

## Diverse Baselines

Don't commit to one model type early. Train multiple frameworks to understand your data:

```python
# Quick baseline comparison across frameworks
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

baselines = {
    'xgb': XGBRegressor(device='cuda', n_estimators=1000, learning_rate=0.05),
    'lgb': LGBMRegressor(device='gpu', n_estimators=1000, learning_rate=0.05),
    'cat': CatBoostRegressor(task_type='GPU', iterations=1000, learning_rate=0.05, verbose=0),
}

for name, model in baselines.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f"{name}: RMSE = {np.sqrt(-scores.mean()):.4f} (+/- {np.sqrt(-scores).std():.4f})")
```

**Why this matters:** Different model types reveal different aspects of your data. If linear models do surprisingly well, interactions may not matter. If one GBDT framework dominates, that guides where to focus.

## Workflow Overview

1. **Smarter EDA** - Check train/test distribution shift, temporal patterns in target
2. **Diverse Baselines** - Compare CatBoost, XGBoost, LightGBM, neural nets
3. **Setup Era-Based CV** - GroupKFold by era (no temporal leakage)
4. **Feature Engineering** - Interactions, aggregations, combinations (GPU-accelerated)
5. **Train Deep Models** - Multiple frameworks × multiple targets × multiple seeds
6. **Ensemble: Hill Climbing** - Optimize weights on OOF predictions
7. **Ensemble: Stacking** - Meta-model on OOF if beneficial
8. **Pseudo-Labeling** - Expand training with confident predictions (if unlabeled data available)
9. **Extra Training** - Seed averaging + retrain on 100% data
10. **Evaluate Payout** - numerai_corr + 3× correlation_contribution

## Quick Start

### Installation
```bash
# RAPIDS suite (cuDF, cuML) - requires CUDA 12.x
pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# Gradient boosting frameworks with GPU
pip install xgboost catboost lightgbm

# Optional: For neural approaches
pip install pytorch-tabnet tabpfn
```

### GPU-Accelerated Data Loading
```python
import cudf

# Load data on GPU (150x faster than pandas for large files)
train = cudf.read_parquet("train.parquet")
test = cudf.read_parquet("test.parquet")

# Or use pandas accelerator mode (zero code change)
# %load_ext cudf.pandas
# import pandas as pd  # now GPU-accelerated
```

### Baseline Model Template
```python
import xgboost as xgb
import numpy as np

# Start with LIGHT parameters, use early stopping
params = {
    "device": "cuda",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "learning_rate": 0.01,       # Start higher
    "max_depth": 8,
    "n_estimators": 2000,        # Start smaller
    "colsample_bytree": 0.1,
    "subsample": 0.8,
    "early_stopping_rounds": 100,
}

# Time-ordered CV with embargo (NOT GroupKFold!)
splits = make_embargo_splits(df['era'], n_splits=5, embargo_eras=4)
oof_preds = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    oof_preds[val_idx] = model.predict(X_val)
    
    # Save checkpoint
    joblib.dump(model, f'xgb_fold{fold}.pkl')
```

## Reference Materials

For detailed guidance on specific topics, see:

- **[references/numerai_guide.md](references/numerai_guide.md)**: Numerai competition specifics - era validation, payout scoring, multi-target ensembling
- **[references/gbdt_frameworks.md](references/gbdt_frameworks.md)**: XGBoost vs LightGBM vs CatBoost selection and GPU configs
- **[references/validation_strategies.md](references/validation_strategies.md)**: Cross-validation strategies and leakage prevention
- **[references/feature_engineering.md](references/feature_engineering.md)**: GPU-accelerated feature generation patterns
- **[references/ensembling.md](references/ensembling.md)**: Hill climbing, stacking, and pseudo-labeling
- **[references/gpu_acceleration.md](references/gpu_acceleration.md)**: RAPIDS cuDF/cuML optimization guide
- **[references/neural_approaches.md](references/neural_approaches.md)**: TabNet, TabPFN, and deep learning for tabular

## Scripts

- **[scripts/numerai_trainer.py](scripts/numerai_trainer.py)**: Numerai-optimized deep model training with era-based CV
- **[scripts/baseline_trainer.py](scripts/baseline_trainer.py)**: Multi-framework baseline training
- **[scripts/cv_validator.py](scripts/cv_validator.py)**: Cross-validation with proper OOF generation
- **[scripts/hill_climb_ensemble.py](scripts/hill_climb_ensemble.py)**: GPU-accelerated ensemble optimization

## Deep Model Configuration

**This dataset rewards DEEP models.** But start light and let early stopping guide you deeper.

### Progressive Training Strategy
1. **Start light**: n_estimators=2,000, learning_rate=0.01
2. **Use early stopping** to find optimal tree count
3. **Gradually deepen**: Reduce LR, increase trees based on findings
4. **Target**: n_estimators=30,000, learning_rate=0.001 for final models

### GPU Availability
| Framework | GPU Support | Device Setting |
|-----------|-------------|----------------|
| XGBoost | ✅ CUDA | `device="cuda"` |
| CatBoost | ✅ GPU | `task_type="GPU"` |
| LightGBM | ❌ CPU only | Keep smaller models (2K trees) |

**Strategy**: Let XGBoost and CatBoost bring deep nuance (30K trees). Use LightGBM with lighter params for ensemble diversity.

### Light vs Deep Parameters

| Parameter | Light (Start Here) | Deep (Target) |
|-----------|-------------------|---------------|
| n_estimators | **2,000** | 30,000 |
| learning_rate | **0.01** | 0.001 |
| max_depth | 5 | **10** |
| num_leaves (LGB) | 15 | 1,024 |
| min_data_in_leaf | 1,000 | **10,000** |
| early_stopping | **100 rounds** | 500 rounds |

### XGBoost Deep Config (GPU)
```python
params = {
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
    "early_stopping_rounds": 500,
}
```

### CatBoost Deep Config (GPU)
```python
params = {
    "task_type": "GPU",
    "devices": "0",
    "iterations": 30_000,
    "learning_rate": 0.001,
    "depth": 10,
    "min_data_in_leaf": 10_000,
    "rsm": 0.1,
    "subsample": 0.8,
    "early_stopping_rounds": 500,
}
```

### LightGBM Config (CPU - Keep Light)
```python
params = {
    "device": "cpu",  # GPU not supported on this platform
    "objective": "regression",
    "n_estimators": 2_000,  # Keep smaller for CPU
    "learning_rate": 0.01,
    "max_depth": 8,
    "num_leaves": 255,
    "min_data_in_leaf": 1_000,
    "feature_fraction": 0.1,
    "bagging_fraction": 0.8,
    "early_stopping_rounds": 100,
}
```

## Numerai-Style Workflow

For datasets with era-based structure and custom scoring:

### 1. Time-Ordered CV with Embargo (CRITICAL)
```python
# WARNING: GroupKFold causes severe temporal leakage!
# Adjacent eras have ~99% target correlation (overlapping forward returns)
# Must use time-ordered CV with embargo gap

def make_embargo_splits(
    era: pd.Series, 
    n_splits: int = 5, 
    embargo_eras: int = 4,
    min_train_ratio: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Time-ordered CV with embargo between train and val.
    - Training data always comes BEFORE validation data temporally
    - Embargo gap prevents target leakage from overlapping returns
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

# Usage
splits = make_embargo_splits(df['era'], n_splits=5, embargo_eras=4)
for train_idx, val_idx in splits:
    # Train always comes BEFORE val temporally
    pass
```

### 2. Multi-Target Training
Train separate models on auxiliary targets for ensemble diversity:
```python
targets = ["target_ender_20", "target_ender_60", "target_jasper_20", ...]
oof_preds = {}
for target in targets:
    oof_preds[target] = train_model(X, df[target], groups=df['era'])
```

### 3. Rank-Gaussian Ensembling
```python
from scipy import stats

def rank_gauss(s):
    """Rank-normalize and gaussianize."""
    ranks = s.rank(method='average') / (len(s) + 1)
    return stats.norm.ppf(ranks) / stats.norm.ppf(ranks).std()

# Apply per-era, then combine
for col in pred_cols:
    df[col] = df.groupby('era')[col].transform(rank_gauss)
final = df[pred_cols].dot(weights)
final = df.groupby('era')['final'].transform(rank_gauss)
```

### 4. Payout Scoring
```python
from numerai_tools.scoring import numerai_corr, correlation_contribution

# IMPORTANT: Compute per-era, then average
# numerai_corr(predictions: DataFrame, targets: Series)
# correlation_contribution(predictions: DataFrame, meta_model: Series, live_targets: Series)

pred_col = 'prediction'  # or list of prediction columns

per_era_nc = df.groupby('era').apply(
    lambda d: numerai_corr(d[[pred_col]], d['target_ender_20'])
)
per_era_cc = df.groupby('era').apply(
    lambda d: correlation_contribution(d[[pred_col]], d['v52_lgbm_ender20'], d['target_ender_20'])
)

# Mean across eras
nc = per_era_nc[pred_col].mean()
cc = per_era_cc[pred_col].mean()

# Payout = 0.75 * NC + 2.25 * CC (CC weighted 3x more!)
payout = 0.75 * nc + 2.25 * cc
```

## Model Selection Decision Tree

```
For this Numerai-style dataset:

All three frameworks are viable - explore freely:
├─ CatBoost GPU (known strong performer in Numerai)
├─ XGBoost GPU (robust, well-documented)
└─ LightGBM CPU (fast training, ensemble diversity)

Ensemble diversity is key:
└─ Train each framework on all 6 targets
   └─ Different tree-building strategies capture different patterns

Parameter regime:
├─ Exploration: 2K trees, LR=0.01 (quick iteration)
└─ Production: 30K trees, LR=0.001 (GPU models only)

Key insight: Deep GPU models >> shallow models for this data
```

## Memory Management for 4.5M Rows

```python
# Dataset: 4.5M rows × 2748 int8 features
# Memory: 4.5M × 2748 × 1 byte ≈ 12.4 GB (very efficient!)
# A100 80GB + 160GB RAM = abundant resources

# Features are pre-quantized int8 [0,1,2,3,4] - no conversion needed
df = pd.read_parquet("dataset.parquet")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

# Keep int8! Tree models handle it natively
# Do NOT convert to float32 - wastes 4x memory
X = df[feature_cols].values  # Stays as int8

# For GPU training with cuDF
import cudf
gdf = cudf.read_parquet("dataset.parquet")
```

## Competition-Winning Checklist

### 1. Smarter EDA
- [ ] Check train vs test distribution for important features
- [ ] Analyze target variable for temporal patterns across eras
- [ ] Identify data leakage risks

### 2. Data Preparation
- [ ] Load dataset.parquet and benchmark.parquet, merge on id
- [ ] Identify feature columns (start with "feature_")
- [ ] Verify features are int8 [0-4] - do NOT scale or transform
- [ ] Never use "era" as a feature (grouping only)

### 3. Diverse Baselines
- [ ] Train quick baselines with CatBoost, XGBoost, LightGBM
- [ ] Compare performance to understand data characteristics

### 4. Feature Engineering
- [ ] Explore feature interactions (pairwise combinations of strong features)
- [ ] Try aggregation features (mean/std of feature groups per era)
- [ ] Test dimensionality reduction (PCA components as additional features)
- [ ] Use GPU-accelerated generation with cuDF

### 5. Validation Setup
- [ ] Use time-ordered CV with embargo (NOT GroupKFold - causes leakage!)
- [ ] Set embargo_eras=4 for 20-day targets
- [ ] Verify train eras always come BEFORE val eras
- [ ] Evaluate with groupby('era').apply() pattern for numerai_corr and correlation_contribution
- [ ] Track payout = 0.75*NC + 2.25*CC
- [ ] Realistic OOF scores: NC ~0.03-0.055, CC ~0.01-0.035

### 6. Model Training
- [ ] **Start light**: n_estimators=2000, LR=0.01, early_stopping=100
- [ ] Use early stopping to find optimal tree count
- [ ] **GPU models** (XGBoost, CatBoost): Progress to deep params (30K trees)
- [ ] **CPU models** (LightGBM): Keep at 2K trees for efficiency
- [ ] Train on main target: target_ender_20
- [ ] Train on auxiliary targets for ensemble diversity
- [ ] **Save models and OOF predictions after each run**

### 7. Ensembling
- [ ] **Hill climbing**: Optimize weights on OOF predictions
- [ ] **Stacking**: Meta-model on diverse OOF predictions
- [ ] **Multi-target blend**: Rank-gaussianize per era, weighted combine

### 8. Pseudo-Labeling (if unlabeled data available)
- [ ] Generate soft pseudo-labels
- [ ] Multi-round for best results
- [ ] Avoid leakage: separate pseudo-labels per CV fold

### 9. Extra Training
- [ ] **Seed averaging**: Train with 5-10 seeds, average predictions
- [ ] **Retrain on 100% data**: Final model uses all training data

### 10. Final Evaluation
- [ ] Monitor correlation_contribution (3x weight in payout!)
- [ ] Compare ensemble vs single best model
- [ ] Try neural approaches (TabNet) for diversity

## Advanced Techniques

Beyond basic model training, these techniques often separate top performers:

### Feature Engineering
With 2748 features, strategic engineering can unlock hidden signal:
```python
# See references/feature_engineering.md for full patterns
# Example: pairwise interactions of top features
top_features = get_top_n_by_importance(model, n=50)
for f1, f2 in combinations(top_features, 2):
    df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
```

### Hill Climbing Ensemble
Optimize weights across all OOF predictions (targets × frameworks):
```python
# See scripts/hill_climb_ensemble.py
from hill_climb_ensemble import HillClimbingEnsemble

# Collect OOF predictions from all models
oof_matrix = np.column_stack([
    oof_catboost_ender20, oof_xgb_ender20, oof_lgb_ender20,
    oof_catboost_ender60, oof_xgb_ender60, ...
])

hc = HillClimbingEnsemble(metric='numerai_corr')
weights = hc.fit(oof_matrix, y_true)
ensemble_pred = oof_matrix @ weights
```

### Stacking
Train a meta-model on OOF predictions:
```python
# See references/ensembling.md for full stacking guide
# Level 1: Base model OOF predictions become meta-features
meta_features = np.column_stack([oof_cat, oof_xgb, oof_lgb, oof_tabnet])

# Level 2: Meta-model (often simple linear or shallow tree)
from sklearn.linear_model import Ridge
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features[train_idx], y[train_idx])
final_pred = meta_model.predict(meta_features)
```

### Neural Approaches for Diversity
Add TabNet or MLP predictions to ensemble for different inductive bias:
```python
# See references/neural_approaches.md
from pytorch_tabnet.tab_model import TabNetRegressor

tabnet = TabNetRegressor(
    n_d=32, n_a=32, n_steps=5,
    gamma=1.5, lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=50, gamma=0.9),
    mask_type='entmax'
)
```

### Pseudo-Labeling
Turn unlabeled data into training signal (see [references/ensembling.md](references/ensembling.md)):
```python
# Use soft labels (probabilities) for regularization
# Multi-round pseudo-labeling often outperforms single-pass
# Avoid leakage: use k separate pseudo-label sets for k-fold CV
```

### Extra Training (Final Polish)
Two techniques that squeeze out extra performance after optimization:

**Seed Averaging:** Train identical models with different random seeds, average predictions:
```python
# Reduces variance, improves robustness
seeds = [42, 123, 456, 789, 1000]
predictions = []

for seed in seeds:
    model = CatBoostRegressor(**deep_params, random_seed=seed)
    model.fit(X_train, y_train)
    predictions.append(model.predict(X_test))

final_pred = np.mean(predictions, axis=0)
```

**Retrain on 100% Data:** After finding optimal hyperparameters via CV, retrain final model on all data:
```python
# After hyperparameter search with CV
best_params = {...}  # Found via CV

# Final model uses ALL training data
final_model = CatBoostRegressor(**best_params)
final_model.fit(X_full, y_full)  # No holdout
final_predictions = final_model.predict(X_test)
```
