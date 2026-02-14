# GPU Acceleration with RAPIDS

## Overview

RAPIDS is NVIDIA's open-source suite for GPU-accelerated data science:
- **cuDF**: GPU DataFrame library (pandas-like)
- **cuML**: GPU machine learning (scikit-learn-like)
- **cuGraph**: GPU graph analytics
- **cuPy**: GPU numerical computing (NumPy-like)

A100 80GB can achieve 50-150x speedups over CPU for data operations.

## Installation

```bash
# For CUDA 12.x
pip install cudf-cu12 cuml-cu12 cupy-cuda12x dask-cudf-cu12 \
    --extra-index-url=https://pypi.nvidia.com

# Or via conda (recommended for full RAPIDS suite)
conda install -c rapidsai -c conda-forge -c nvidia rapids=25.02 python=3.11 cuda-version=12.4
```

## cuDF: GPU DataFrames

### Zero-Code-Change Acceleration

```python
# Method 1: Magic command in Jupyter
%load_ext cudf.pandas
import pandas as pd  # Now GPU-accelerated!

# Method 2: Environment variable
# Set CUDF_PANDAS=1 before running script
# python my_script.py  # pandas operations now use GPU

# Method 3: Explicit import
import cudf
df = cudf.read_parquet("data.parquet")  # 50-150x faster than pandas
```

### Common Operations

```python
import cudf
import cupy as cp

# Reading data (Parquet is fastest)
df = cudf.read_parquet("train.parquet")
# df = cudf.read_csv("train.csv")  # CSV also supported

# Basic operations (same as pandas)
df.head()
df.info()
df.describe()

# Filtering
filtered = df[df['value'] > 100]
filtered = df.query('value > 100 and category == "A"')

# GroupBy aggregations (massively faster on GPU)
agg = df.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max', 'sum', 'count'],
    'other': ['mean', 'median']
})

# Joins (10-50x faster than pandas)
merged = df1.merge(df2, on='key', how='left')

# Sorting
sorted_df = df.sort_values('value', ascending=False)

# Apply custom functions (use cuPy for GPU operations)
df['log_value'] = cp.log(df['value'].values + 1)
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()

# Value counts
counts = df['category'].value_counts(normalize=True)

# String operations (GPU-accelerated)
df['lower'] = df['text'].str.lower()
df['contains'] = df['text'].str.contains('pattern')
```

### Converting Between cuDF and pandas

```python
# pandas → cuDF
gdf = cudf.DataFrame(pandas_df)

# cuDF → pandas
pandas_df = gdf.to_pandas()

# cuDF → NumPy
numpy_arr = gdf.values.get()  # Note: .get() copies from GPU to CPU

# cuDF → CuPy (stays on GPU)
cupy_arr = gdf.values  # No copy, stays on GPU
```

### Memory Management

```python
import gc
import cupy as cp

# Check GPU memory
mempool = cp.get_default_memory_pool()
print(f"Used: {mempool.used_bytes() / 1e9:.2f} GB")
print(f"Total: {mempool.total_bytes() / 1e9:.2f} GB")

# Free GPU memory
del df
gc.collect()
mempool.free_all_blocks()

# Processing large data in chunks
def process_large_file(path, chunk_size=1_000_000):
    results = []
    for chunk in cudf.read_csv(path, chunksize=chunk_size):
        # Process chunk
        processed = chunk.groupby('key').mean()
        results.append(processed)
        
        # Clear memory
        del chunk
        gc.collect()
        mempool.free_all_blocks()
    
    return cudf.concat(results)
```

## cuML: GPU Machine Learning

### Accelerated Algorithms

```python
from cuml import (
    LogisticRegression, Ridge, Lasso,
    RandomForestClassifier, RandomForestRegressor,
    KNeighborsClassifier, KNeighborsRegressor,
    PCA, UMAP, TSNE,
    KMeans, DBSCAN,
    StandardScaler, MinMaxScaler
)

# All have scikit-learn compatible API
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

### Zero-Code-Change scikit-learn Acceleration

```python
# Set environment before importing sklearn
import os
os.environ['CUML_ACCEL'] = '1'

# Or use magic command
%load_ext cuml.accel

# Now scikit-learn is GPU-accelerated
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# These now run on GPU automatically
scaler = StandardScaler().fit_transform(X)
pca = PCA(n_components=50).fit_transform(X)
rf = RandomForestClassifier().fit(X_train, y_train)
```

### cuML with Gradient Boosting

cuML integrates with GPU-accelerated gradient boosting:

```python
# XGBoost with cuML preprocessing
import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler
import xgboost as xgb

# Load and preprocess with cuDF/cuML
df = cudf.read_parquet("train.parquet")
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DMatrix for XGBoost (stays on GPU)
dtrain = xgb.DMatrix(X_scaled.values, label=y.values)

# Train XGBoost on GPU
params = {
    'device': 'cuda',
    'tree_method': 'hist',
    'objective': 'reg:squarederror'
}
model = xgb.train(params, dtrain, num_boost_round=500)
```

## cuPy: GPU NumPy

```python
import cupy as cp

# Create arrays on GPU
x = cp.array([1, 2, 3, 4, 5])
y = cp.arange(1000000)
z = cp.random.randn(1000, 1000)

# Mathematical operations (GPU-accelerated)
result = cp.sqrt(cp.sum(z ** 2, axis=1))

# Linear algebra
eigenvalues = cp.linalg.eigvalsh(z @ z.T)
inv = cp.linalg.inv(z)

# FFT
fft_result = cp.fft.fft(y)

# Custom kernels for maximum performance
kernel = cp.ElementwiseKernel(
    'float32 x, float32 y',
    'float32 z',
    'z = sqrt(x * x + y * y)',
    'my_custom_kernel'
)
z = kernel(x, y)
```

## Dask-cuDF: Distributed GPU DataFrames

For datasets larger than GPU memory:

```python
import dask_cudf

# Read large dataset distributed across GPU
ddf = dask_cudf.read_parquet("large_data/*.parquet")

# Operations are lazy (computed on demand)
result = ddf.groupby('key').agg({'value': 'mean'})

# Compute to materialize
result_df = result.compute()

# With multiple GPUs
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()
client = Client(cluster)

# Now operations use all available GPUs
```

## Performance Optimization Tips

### 1. Minimize CPU-GPU Transfers

```python
# BAD: Multiple transfers
for col in columns:
    df[col] = pandas_func(df[col].to_pandas())

# GOOD: Process entirely on GPU
for col in columns:
    df[col] = cudf_func(df[col])
```

### 2. Use Appropriate Data Types

```python
# Use float32 instead of float64 (2x memory savings)
df = df.astype({col: 'float32' for col in numeric_cols})

# Use int32 for categorical indices
df['cat_encoded'] = df['category'].astype('category').cat.codes.astype('int32')
```

### 3. Batch Operations

```python
# BAD: Many small operations
for i in range(len(df)):
    df.iloc[i] = process(df.iloc[i])

# GOOD: Vectorized operations
df = process_vectorized(df)
```

### 4. Use Parquet for I/O

```python
# Parquet is fastest for GPU loading
df.to_parquet("data.parquet")
df = cudf.read_parquet("data.parquet")

# With compression
df.to_parquet("data.parquet", compression='snappy')
```

### 5. Profile GPU Usage

```python
# Monitor GPU memory
import cupy as cp

def memory_stats():
    pool = cp.get_default_memory_pool()
    return {
        'used_gb': pool.used_bytes() / 1e9,
        'total_gb': pool.total_bytes() / 1e9,
        'utilization': pool.used_bytes() / pool.total_bytes() * 100
    }

print(memory_stats())

# Use nvidia-smi in terminal
# nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1
```

## Complete Pipeline Example

```python
import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
import xgboost as xgb
import gc

# 1. Load data on GPU
train = cudf.read_parquet("train.parquet")
test = cudf.read_parquet("test.parquet")

# 2. Feature engineering with cuDF
for c1, c2 in combinations(cat_cols, 2):
    train[f'{c1}_{c2}'] = train[c1].astype(str) + "_" + train[c2].astype(str)
    test[f'{c1}_{c2}'] = test[c1].astype(str) + "_" + test[c2].astype(str)

# 3. Preprocessing with cuML
X_train = train.drop('target', axis=1)
y_train = train['target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test)

# 4. Optional: Dimensionality reduction
pca = PCA(n_components=500)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 5. Train XGBoost on GPU
dtrain = xgb.DMatrix(X_train_pca.values, label=y_train.values)
dtest = xgb.DMatrix(X_test_pca.values)

params = {
    'device': 'cuda',
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 8
}

model = xgb.train(params, dtrain, num_boost_round=1000)
predictions = model.predict(dtest)

# 6. Clean up GPU memory
del train, test, X_train, X_train_scaled, X_train_pca
gc.collect()
cp.get_default_memory_pool().free_all_blocks()
```
