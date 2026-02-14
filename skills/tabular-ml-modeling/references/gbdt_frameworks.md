# Gradient Boosting Frameworks

## Framework Comparison

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Tree Growth | Level-wise (balanced) | Leaf-wise (deeper) | Symmetric (balanced) |
| Categorical Handling | Requires encoding | Limited native support | Best native handling |
| Training Speed | Moderate | Fastest | Moderate |
| Memory Usage | High | Low | Moderate |
| Hyperparameter Tuning | Extensive needed | Moderate | Minimal needed |
| **GPU on this platform** | ✅ CUDA | ❌ CPU only | ✅ GPU |
| Best For | General purpose | Large datasets | Categorical-heavy |

## GPU Availability Note

On this platform:
- **XGBoost**: Full CUDA GPU support → Use for deep models (30K trees)
- **CatBoost**: Full GPU support → Use for deep models (30K trees)  
- **LightGBM**: CPU only → Keep lighter (2K trees) for ensemble diversity

## XGBoost GPU Configuration

```python
import xgboost as xgb

params = {
    # GPU settings
    "device": "cuda",           # Use GPU
    "tree_method": "hist",      # Histogram-based (required for GPU)
    
    # Core parameters
    "objective": "reg:squarederror",  # Regression
    # "objective": "binary:logistic",  # Binary classification
    # "objective": "multi:softprob",   # Multiclass
    
    # Tree structure
    "max_depth": 8,             # Deeper for complex patterns
    "min_child_weight": 1,      # Regularization
    "subsample": 0.8,           # Row sampling
    "colsample_bytree": 0.8,    # Column sampling
    
    # Learning
    "learning_rate": 0.05,      # Lower for more trees
    "n_estimators": 2000,       # More with early stopping
    "early_stopping_rounds": 50,
    
    # Regularization
    "reg_alpha": 0.1,           # L1
    "reg_lambda": 1.0,          # L2
    
    "random_state": 42,
    "n_jobs": -1,
}

# For 7M rows, consider:
params["max_bin"] = 256  # Default 256, can reduce for memory
```

## LightGBM Configuration (CPU)

**Note:** LightGBM GPU is not supported on this platform. Keep models smaller (2K trees) and let XGBoost/CatBoost handle deep learning.

```python
import lightgbm as lgb

params = {
    # CPU only - GPU not available
    "device": "cpu",
    
    # Core parameters
    "objective": "regression",  # or "binary", "multiclass"
    "metric": "rmse",          # or "auc", "logloss"
    "boosting_type": "gbdt",   # or "dart", "rf"
    
    # Tree structure - keep lighter for CPU
    "num_leaves": 255,         # Main complexity control
    "max_depth": 8,            # Limit depth for CPU efficiency
    "min_data_in_leaf": 1000,  # Prevent overfitting
    
    # Sampling
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.1,   # Low for high-dimensional data
    
    # Learning - lighter for CPU
    "learning_rate": 0.01,
    "n_estimators": 2000,       # Keep smaller for CPU
    "early_stopping_rounds": 100,
    
    # Regularization
    "lambda_l1": 0.0,
    "lambda_l2": 0.1,
    
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}
```

## CatBoost GPU Configuration

```python
from catboost import CatBoostRegressor, CatBoostClassifier

params = {
    # GPU settings
    "task_type": "GPU",
    "devices": "0",  # GPU device ID
    
    # Core parameters
    "loss_function": "RMSE",  # or "Logloss", "MultiClass"
    "eval_metric": "RMSE",
    
    # Tree structure - CatBoost uses depth, not leaves
    "depth": 8,               # Symmetric trees
    "min_data_in_leaf": 1,
    
    # Learning
    "learning_rate": 0.05,
    "iterations": 2000,
    "early_stopping_rounds": 50,
    
    # Regularization
    "l2_leaf_reg": 3.0,       # L2 regularization
    "random_strength": 1.0,   # Randomness for scoring
    "bagging_temperature": 1.0,
    
    # CatBoost specific
    "border_count": 254,      # Like max_bin
    "grow_policy": "SymmetricTree",  # or "Depthwise", "Lossguide"
    
    "random_seed": 42,
    "verbose": 100,
}

# Categorical features (native handling - major advantage)
cat_features = [col for col in X.columns if X[col].dtype == 'object']
model = CatBoostRegressor(**params)
model.fit(X_train, y_train, cat_features=cat_features, 
          eval_set=(X_val, y_val))
```

## GPU Memory Considerations

For 7M rows × 2750 features:

```python
# XGBoost: Uses GPU memory efficiently with histogram method
# - max_bin=256: ~40GB GPU memory
# - max_bin=128: ~30GB GPU memory

# LightGBM: Most memory efficient
# - Leaf-wise reduces memory needs
# - gpu_use_dp=False keeps float32

# CatBoost: Moderate memory usage
# - border_count affects memory
# - Can use CTR caching for categoricals

# General tips:
# 1. Use float32 not float64
# 2. Reduce max_bin if memory constrained
# 3. Use subsample/bagging to reduce per-tree memory
# 4. Monitor with nvidia-smi during training
```

## Hyperparameter Tuning Priorities

### XGBoost (in order)
1. `max_depth` (4-12)
2. `learning_rate` (0.01-0.3)
3. `subsample` (0.6-1.0)
4. `colsample_bytree` (0.6-1.0)
5. `reg_alpha`, `reg_lambda`

### LightGBM (in order)
1. `num_leaves` (31-512)
2. `learning_rate` (0.01-0.3)
3. `feature_fraction` (0.6-1.0)
4. `bagging_fraction` (0.6-1.0)
5. `min_data_in_leaf` (10-100)

### CatBoost (in order)
1. `depth` (4-10)
2. `learning_rate` (0.01-0.3)
3. `l2_leaf_reg` (1-10)
4. `random_strength` (0.5-2)
5. `bagging_temperature` (0-1)

## When to Use Each Framework

**XGBoost**: Strong community, well-documented, robust GPU support. Level-wise tree growth provides good regularization.

**LightGBM**: Fast training speed, memory efficient. Leaf-wise growth can find complex patterns. Most common in Kaggle.

**CatBoost**: Strong default performance with minimal tuning. Symmetric trees provide regularization. Known for strong results in Numerai tournaments. Fast inference (30-60x faster prediction than XGB/LGB).

**Recommendation**: Try all three frameworks - different tree-building strategies capture different patterns. Ensemble diversity often beats single-framework optimization.
