# Feature Engineering

## GPU-Accelerated Feature Engineering with cuDF

For 7M rows × 2750 features, GPU acceleration is essential.

### Setup

```python
import cudf
import cupy as cp
import numpy as np

# Load data to GPU
train = cudf.read_parquet("train.parquet")
test = cudf.read_parquet("test.parquet")

# Or convert from pandas
train = cudf.DataFrame(train_pd)
```

## Categorical Feature Engineering

### Categorical Combinations (Interaction Features)

```python
def create_cat_combinations(df, cat_cols):
    """Create pairwise categorical combinations."""
    new_features = {}
    
    for i, c1 in enumerate(cat_cols[:-1]):
        for c2 in cat_cols[i+1:]:
            name = f"{c1}_{c2}"
            # String concatenation on GPU
            new_features[name] = (
                df[c1].astype(str) + "_" + df[c2].astype(str)
            )
    
    return cudf.DataFrame(new_features)

# Example: 8 categorical cols → 28 new features
cat_cols = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
combo_features = create_cat_combinations(train, cat_cols)
```

### Frequency Encoding

```python
def frequency_encode(df, col):
    """Replace category with its frequency."""
    freq = df[col].value_counts(normalize=True)
    return df[col].map(freq)

# Apply to all categorical columns
for col in cat_cols:
    train[f'{col}_freq'] = frequency_encode(train, col)
    test[f'{col}_freq'] = train[col].value_counts(normalize=True).reindex(test[col]).values
```

### Target Encoding (with CV to prevent leakage)

```python
def target_encode_cv(train_df, test_df, cat_col, target_col, n_splits=5):
    """Target encoding with cross-validation to prevent leakage."""
    from sklearn.model_selection import KFold
    
    train_encoded = cp.zeros(len(train_df))
    global_mean = train_df[target_col].mean()
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = cp.arange(len(train_df))
    
    for train_idx, val_idx in kfold.split(X.get()):
        # Calculate means from training fold only
        train_fold = train_df.iloc[train_idx]
        means = train_fold.groupby(cat_col)[target_col].mean()
        
        # Map to validation fold
        val_cats = train_df.iloc[val_idx][cat_col]
        train_encoded[val_idx] = val_cats.map(means).fillna(global_mean).values
    
    # For test: use all training data
    full_means = train_df.groupby(cat_col)[target_col].mean()
    test_encoded = test_df[cat_col].map(full_means).fillna(global_mean)
    
    return train_encoded, test_encoded
```

## Numerical Feature Engineering

### Statistical Aggregations

```python
def create_agg_features(df, group_col, agg_cols):
    """Create aggregation features grouped by a column."""
    agg_funcs = ['mean', 'std', 'min', 'max', 'sum', 'count']
    
    for agg_col in agg_cols:
        aggs = df.groupby(group_col)[agg_col].agg(agg_funcs)
        aggs.columns = [f'{group_col}_{agg_col}_{func}' for func in agg_funcs]
        df = df.merge(aggs, on=group_col, how='left')
    
    return df

# Example: Group statistics
train = create_agg_features(train, 'user_id', ['amount', 'duration'])
```

### Polynomial and Interaction Features

```python
def create_polynomial_features(df, num_cols, degree=2):
    """Create polynomial features for numerical columns."""
    new_features = {}
    
    # Squares
    if degree >= 2:
        for col in num_cols:
            new_features[f'{col}_sq'] = df[col] ** 2
    
    # Interactions (pairwise products)
    for i, c1 in enumerate(num_cols[:-1]):
        for c2 in num_cols[i+1:]:
            new_features[f'{c1}_x_{c2}'] = df[c1] * df[c2]
    
    return cudf.DataFrame(new_features)
```

### Ratio and Difference Features

```python
def create_ratio_features(df, num_cols):
    """Create ratio features between numerical columns."""
    new_features = {}
    
    for i, c1 in enumerate(num_cols[:-1]):
        for c2 in num_cols[i+1:]:
            # Ratio (with epsilon to avoid division by zero)
            eps = 1e-8
            new_features[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + eps)
            new_features[f'{c2}_div_{c1}'] = df[c2] / (df[c1] + eps)
            
            # Difference
            new_features[f'{c1}_minus_{c2}'] = df[c1] - df[c2]
    
    return cudf.DataFrame(new_features)
```

### Binning Numerical Features

```python
def create_bins(df, col, n_bins=10):
    """Bin numerical column into categories."""
    # Quantile-based binning
    quantiles = df[col].quantile(cp.linspace(0, 1, n_bins + 1).get())
    return cudf.cut(df[col], bins=quantiles.to_pandas(), labels=False)

# Or use cuML for more sophisticated binning
from cuml.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
df['col_binned'] = discretizer.fit_transform(df[['col']])
```

## Time-Based Features

```python
def create_time_features(df, date_col):
    """Extract time-based features from datetime column."""
    df = df.copy()
    dt = df[date_col]
    
    df[f'{date_col}_year'] = dt.dt.year
    df[f'{date_col}_month'] = dt.dt.month
    df[f'{date_col}_day'] = dt.dt.day
    df[f'{date_col}_dayofweek'] = dt.dt.dayofweek
    df[f'{date_col}_hour'] = dt.dt.hour
    df[f'{date_col}_is_weekend'] = dt.dt.dayofweek >= 5
    df[f'{date_col}_quarter'] = dt.dt.quarter
    
    # Cyclical encoding
    df[f'{date_col}_month_sin'] = cp.sin(2 * cp.pi * dt.dt.month / 12)
    df[f'{date_col}_month_cos'] = cp.cos(2 * cp.pi * dt.dt.month / 12)
    df[f'{date_col}_dow_sin'] = cp.sin(2 * cp.pi * dt.dt.dayofweek / 7)
    df[f'{date_col}_dow_cos'] = cp.cos(2 * cp.pi * dt.dt.dayofweek / 7)
    
    return df
```

## Memory-Efficient Feature Generation

For 7M rows, generate features in batches:

```python
def batch_feature_engineering(df, feature_func, batch_size=1_000_000):
    """Apply feature engineering in batches to manage memory."""
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_features = feature_func(batch)
        results.append(batch_features)
        
        # Clear GPU memory
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    
    return cudf.concat(results, ignore_index=True)
```

## Feature Selection

### Importance-Based Selection

```python
def select_by_importance(model, X, threshold=0.001):
    """Select features above importance threshold."""
    importance = model.feature_importances_
    mask = importance > threshold
    selected_cols = X.columns[mask].tolist()
    return selected_cols

# After training initial model
selected_features = select_by_importance(xgb_model, X_train)
print(f"Selected {len(selected_features)} of {X_train.shape[1]} features")
```

### Correlation-Based Removal

```python
def remove_correlated_features(df, threshold=0.95):
    """Remove highly correlated features."""
    corr_matrix = df.corr().abs()
    
    # Upper triangle
    upper = corr_matrix.where(
        cp.triu(cp.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find columns with correlation above threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    return df.drop(columns=to_drop)
```

### Null Importance Test

```python
def null_importance_selection(X, y, model_cls, n_runs=10):
    """Select features that beat random permutation."""
    from sklearn.model_selection import cross_val_score
    
    # Real importance
    model = model_cls()
    model.fit(X, y)
    real_importance = model.feature_importances_
    
    # Null importance (permuted target)
    null_importances = []
    for _ in range(n_runs):
        y_perm = np.random.permutation(y)
        model.fit(X, y_perm)
        null_importances.append(model.feature_importances_)
    
    null_importances = np.array(null_importances)
    
    # Select features where real > 95th percentile of null
    thresholds = np.percentile(null_importances, 95, axis=0)
    selected = X.columns[real_importance > thresholds].tolist()
    
    return selected
```

## Feature Engineering Pipeline

```python
class FeatureEngineer:
    """Complete feature engineering pipeline with GPU acceleration."""
    
    def __init__(self, cat_cols, num_cols, date_cols=None):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.date_cols = date_cols or []
        self.encoders = {}
    
    def fit_transform(self, df, target=None):
        """Fit on training data and transform."""
        df = cudf.DataFrame(df) if not isinstance(df, cudf.DataFrame) else df
        
        # Categorical combinations
        cat_combos = create_cat_combinations(df, self.cat_cols)
        df = cudf.concat([df, cat_combos], axis=1)
        
        # Frequency encoding
        for col in self.cat_cols:
            self.encoders[f'{col}_freq'] = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = frequency_encode(df, col)
        
        # Target encoding (if target provided)
        if target is not None:
            for col in self.cat_cols:
                # Store for test set
                self.encoders[f'{col}_target'] = df.groupby(col)[target].mean()
        
        # Numerical features
        df = cudf.concat([df, create_polynomial_features(df, self.num_cols)], axis=1)
        df = cudf.concat([df, create_ratio_features(df, self.num_cols)], axis=1)
        
        # Time features
        for date_col in self.date_cols:
            df = create_time_features(df, date_col)
        
        return df
    
    def transform(self, df):
        """Transform test data using fitted encoders."""
        df = cudf.DataFrame(df) if not isinstance(df, cudf.DataFrame) else df
        
        # Apply saved encoders
        for col in self.cat_cols:
            df[f'{col}_freq'] = df[col].map(self.encoders[f'{col}_freq'])
            if f'{col}_target' in self.encoders:
                df[f'{col}_target'] = df[col].map(self.encoders[f'{col}_target'])
        
        # Repeat other transformations
        cat_combos = create_cat_combinations(df, self.cat_cols)
        df = cudf.concat([df, cat_combos], axis=1)
        df = cudf.concat([df, create_polynomial_features(df, self.num_cols)], axis=1)
        df = cudf.concat([df, create_ratio_features(df, self.num_cols)], axis=1)
        
        for date_col in self.date_cols:
            df = create_time_features(df, date_col)
        
        return df
```
