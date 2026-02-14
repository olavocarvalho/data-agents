# Validation Strategies

## ⚠️ CRITICAL: Temporal Leakage Warning

**For Numerai and time-series data, GroupKFold causes severe leakage!**

GroupKFold splits data by group but doesn't respect temporal order. When adjacent time periods have high target correlation (e.g., 99.5% for overlapping returns), this causes:
- Inflated OOF scores (~10x higher than reality)
- Models that appear excellent but fail in production

**Solution: Time-ordered CV with embargo gap**

```python
def make_embargo_splits(era, n_splits=5, embargo_eras=4, min_train_ratio=0.5):
    """
    Time-ordered CV with embargo between train and val.
    Train data ALWAYS comes before val data temporally.
    """
    unique_eras = np.sort(era.unique())
    n_eras = len(unique_eras)
    min_train_eras = int(n_eras * min_train_ratio)
    remaining = n_eras - min_train_eras - embargo_eras
    test_size = remaining // n_splits
    era_to_idx = {e: np.where(era == e)[0] for e in unique_eras}
    
    splits = []
    for i in range(n_splits):
        train_end = min_train_eras + (i * test_size)
        test_start = train_end + embargo_eras  # Gap!
        test_end = min(test_start + test_size, n_eras)
        if test_start >= n_eras: break
        
        train_idx = np.concatenate([era_to_idx[e] for e in unique_eras[:train_end]])
        test_idx = np.concatenate([era_to_idx[e] for e in unique_eras[test_start:test_end]])
        splits.append((np.sort(train_idx), np.sort(test_idx)))
    return splits

# Usage
splits = make_embargo_splits(df['era'], n_splits=5, embargo_eras=4)
for train_idx, val_idx in splits:
    assert df.iloc[train_idx]['era'].max() < df.iloc[val_idx]['era'].min()
```

**Embargo sizing:** For 20-day forward returns with weekly eras, use `embargo_eras=4` (~75% overlap requires 4 era gap).

---

## Why Cross-Validation Matters

Single train/test split is unreliable. CV provides:
- More accurate performance estimates
- Detection of overfitting
- Stable out-of-fold (OOF) predictions for stacking
- Better hyperparameter selection

## Cross-Validation Strategies

### Standard K-Fold (Regression)

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = train_model(X_train, y_train, X_val, y_val)
    models.append(model)
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kfold.n_splits

cv_score = mean_squared_error(y, oof_preds, squared=False)
```

### Stratified K-Fold (Classification)

```python
from sklearn.model_selection import StratifiedKFold

# For binary or multiclass classification
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
    # Preserves class distribution in each fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # ...
```

### Time Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

# When data has temporal ordering
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    # train_idx always before val_idx in time
    # Prevents future data leakage
    X_train, X_val = X[train_idx], X[val_idx]
    # ...

# Custom time-based split with gap
def time_based_split(df, date_col, n_splits=5, gap_days=7):
    dates = df[date_col].unique()
    dates = np.sort(dates)
    
    split_size = len(dates) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = dates[split_size * (i + 1)]
        val_start = train_end + pd.Timedelta(days=gap_days)
        val_end = dates[min(split_size * (i + 2), len(dates) - 1)]
        
        train_mask = df[date_col] <= train_end
        val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)
        
        yield df[train_mask].index, df[val_mask].index
```

### Group K-Fold

```python
from sklearn.model_selection import GroupKFold

# When data has groups that shouldn't be split (e.g., users, sessions)
groups = df['user_id'].values
gkfold = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkfold.split(X, y, groups)):
    # All rows from a user are either in train OR val, never both
    X_train, X_val = X[train_idx], X[val_idx]
    # ...

# Stratified Group K-Fold (preserves class balance within groups)
from sklearn.model_selection import StratifiedGroupKFold
sgkfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

## Choosing the Right Strategy

```
Data has temporal ordering?
├─ Yes → TimeSeriesSplit or custom time split
└─ No
   └─ Data has groups (users, sessions)?
      ├─ Yes → GroupKFold or StratifiedGroupKFold
      └─ No
         └─ Classification task?
            ├─ Yes → StratifiedKFold
            └─ No → KFold
```

## Avoiding Data Leakage

### Feature Engineering Leakage

```python
# WRONG: Fit encoder on all data before split
encoder = TargetEncoder().fit(X, y)  # Leaks validation info
X_encoded = encoder.transform(X)

# CORRECT: Fit encoder only on training data within each fold
for train_idx, val_idx in kfold.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]
    
    encoder = TargetEncoder().fit(X_train, y_train)
    X_train_enc = encoder.transform(X_train)
    X_val_enc = encoder.transform(X_val)
```

### Target Encoding Without Leakage

```python
def target_encode_with_cv(X, y, cat_cols, n_splits=5):
    """Target encoding that prevents leakage."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_encoded = X.copy()
    
    for col in cat_cols:
        encoded_col = np.zeros(len(X))
        
        for train_idx, val_idx in kfold.split(X):
            # Calculate means only from training fold
            means = X.iloc[train_idx].groupby(col)[target_col].mean()
            global_mean = y.iloc[train_idx].mean()
            
            # Apply to validation fold
            encoded_col[val_idx] = X.iloc[val_idx][col].map(means)
            encoded_col[val_idx] = encoded_col[val_idx].fillna(global_mean)
        
        X_encoded[f'{col}_target_enc'] = encoded_col
    
    return X_encoded
```

### Scaling/Normalization

```python
from sklearn.preprocessing import StandardScaler

# WRONG: Fit on all data
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# CORRECT: Fit only on training data
for train_idx, val_idx in kfold.split(X, y):
    scaler = StandardScaler().fit(X[train_idx])
    X_train_scaled = scaler.transform(X[train_idx])
    X_val_scaled = scaler.transform(X[val_idx])
```

## CV Score vs Test Score Correlation

Good correlation indicates trustworthy CV:

```python
def analyze_cv_test_correlation(cv_scores, test_scores):
    """Check if CV scores correlate with test scores across experiments."""
    correlation = np.corrcoef(cv_scores, test_scores)[0, 1]
    
    if correlation > 0.8:
        print("Strong correlation - CV is reliable")
    elif correlation > 0.5:
        print("Moderate correlation - CV somewhat reliable")
    else:
        print("Weak correlation - Reconsider CV strategy")
    
    return correlation
```

## Number of Folds

- **5 folds**: Good default, balances bias/variance
- **10 folds**: More stable estimates, 2x training time
- **3 folds**: For very large datasets (7M+ rows)
- **Leave-one-out**: Only for very small datasets

```python
# For 7M rows, 5-fold is usually sufficient
# Each fold trains on 5.6M rows, validates on 1.4M
# 3-fold if training time is prohibitive
```

## Repeated Cross-Validation

For more stable estimates at the cost of computation:

```python
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

# 5 folds × 3 repeats = 15 total fits
rkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

all_scores = []
for train_idx, val_idx in rkfold.split(X):
    # Train and evaluate
    score = evaluate_model(train_idx, val_idx)
    all_scores.append(score)

mean_score = np.mean(all_scores)
std_score = np.std(all_scores)
print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
```
