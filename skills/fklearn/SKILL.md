---
name: fklearn
description: Train, evaluate, and validate ML models using Nubank's fklearn library. Use when building ML pipelines with curried learners, splitting data with space_time_split_dataset, feature transformations (encoding, scaling, imputation), evaluating with ROC-AUC/PR-AUC, tuning hyperparameters, or working with fklearn's functional pattern (learner functions, build_pipeline, validator, combined_evaluators).
---

# fklearn — Nubank's Functional ML Library

fklearn is Nubank's open-source ML library built on functional programming principles. Every learner is a **curried function** that returns `(predict_fn, transformed_df, log)`.

## Core Pattern

All fklearn learners follow the same contract:

```python
# 1. Curry the learner (partial application — all params except df)
learner_fn = lgbm_classification_learner(
    features=FEATURES, target=TARGET, learning_rate=0.1, num_estimators=100
)

# 2. Call with data → returns (predict_fn, predicted_df, log)
predict_fn, train_preds, log = learner_fn(train_df)

# 3. Use predict_fn on new data
test_preds = predict_fn(test_df)
```

**Return types:**
- `predict_fn`: `DataFrame -> DataFrame` — adds prediction column(s)
- `transformed_df`: training data with predictions applied
- `log`: dict with two top-level keys:
  - `"learner_name"`: dict with `features`, `target`, `prediction_column`, `package`, `package_version`, `parameters`, `feature_importance`, `training_samples`
  - `"object"`: the trained model (sibling key, not inside the learner dict)

## Quick Start — Classification Pipeline

```python
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from common_models_storage import load_dataframe
from fklearn.preprocessing.splitting import space_time_split_dataset
from fklearn.training.transformation import label_categorizer, onehot_categorizer
from fklearn.training.classification import lgbm_classification_learner
from fklearn.training.pipeline import build_pipeline
from fklearn.validation.evaluators import roc_auc_evaluator, pr_auc_evaluator, combined_evaluators

# Load data
df = load_dataframe("s3://bucket/data.parquet", nthreads=1)

# Split: returns (train, out_of_space, out_of_time, out_of_space_time)
split_fn = space_time_split_dataset(
    train_start_date="2020-01-01", train_end_date="2020-06-01",
    holdout_end_date="2020-09-01", split_seed=42,
    space_holdout_percentage=0.2, space_column="entity_id", time_column="date"
)
train, oos, oot, oost = split_fn(df)

# Build pipeline: transformations + learner
pipeline = build_pipeline(
    label_categorizer(columns_to_categorize=["cat_col1", "cat_col2"]),
    onehot_categorizer(columns_to_categorize=["multi_cat_col"]),
    lgbm_classification_learner(
        features=["feat1", "feat2", "cat_col1", "cat_col2", "multi_cat_col"],
        target="target",
        learning_rate=0.1,
        num_estimators=100,
        extra_params={"num_leaves": 31, "seed": 42, "verbose": -1},
    ),
)

# Train
predict_fn, train_preds, logs = pipeline(train)

# Evaluate on holdouts
eval_fn = combined_evaluators(evaluators=[
    roc_auc_evaluator(prediction_column="prediction", target_column="target"),
    pr_auc_evaluator(prediction_column="prediction", target_column="target"),
])

results = {name: eval_fn(predict_fn(ds)) for name, ds in
           {"oos": oos, "oot": oot, "oost": oost}.items() if len(ds) > 0}
pd.DataFrame(results).T
```

## Quick Start — Regression Pipeline

```python
from fklearn.training.regression import lgbm_regression_learner
from fklearn.training.transformation import capper
from fklearn.validation.evaluators import r2_evaluator, mse_evaluator

pipeline = build_pipeline(
    capper(columns_to_cap=["income"], precomputed_caps={"income": 100000}),
    lgbm_regression_learner(
        features=["income", "age"], target="spend",
        learning_rate=0.1, num_estimators=200,
    ),
)
predict_fn, train_preds, logs = pipeline(train)
eval_fn = combined_evaluators(evaluators=[
    r2_evaluator(target_column="spend"),
    mse_evaluator(target_column="spend"),
])
```

## Quick Start — Cross-Validation

```python
from fklearn.validation.validator import validator
from fklearn.validation.splitters import k_fold_splitter, stability_curve_time_splitter

# K-Fold CV
cv_results = validator(
    train_data=df,
    split_fn=k_fold_splitter(n_splits=5, random_state=42),
    train_fn=pipeline,
    eval_fn=eval_fn,
)

# Stability curve (temporal degradation)
stability_results = validator(
    train_data=df,
    split_fn=stability_curve_time_splitter(
        training_time_limit="2020-06-01", time_column="date", freq="M"
    ),
    train_fn=pipeline,
    eval_fn=eval_fn,
)

# Robustness: use perturbators to stress-test
from fklearn.validation.perturbators import perturbator, nullify
robust_results = validator(
    train_data=df,
    split_fn=k_fold_splitter(n_splits=3, random_state=42),
    train_fn=pipeline,
    eval_fn=eval_fn,
    perturb_fn_test=lambda df: perturbator(df, cols=FEATURES, corruption_fn=nullify(perc=0.1)),
)
```

For large datasets, use `parallel_validator` (same API, adds `n_jobs` parameter) for multi-process CV.

## Feature Importance

```python
# From logs after training
pd.DataFrame(
    logs["lgbm_classification_learner"]["feature_importance"],
    index=["importance"]
).T.sort_values("importance", ascending=False)
```

Log key is always `learner_function_name` (e.g., `lgbm_classification_learner`, `xgb_regression_learner`).

## Hyperparameter Tuning

```python
from fklearn.tuning.parameter_tuners import random_search_tuner

results = random_search_tuner(
    space={"learning_rate": lambda: np.random.uniform(0.01, 0.3),
           "num_estimators": lambda: np.random.randint(50, 300)},
    train_set=train,
    param_train_fn=lambda params: build_pipeline(
        label_categorizer(columns_to_categorize=cat_cols),
        lgbm_classification_learner(features=FEATURES, target=TARGET, **params),
    ),
    split_fn=k_fold_splitter(n_splits=3),
    eval_fn=eval_fn,
    iterations=20,
)
```

Also available: `grid_search_cv` for exhaustive search. See [references/api.md](references/api.md).

## Available Transformations

Beyond the encoders shown above, fklearn provides: `target_categorizer` (target encoding), `count_categorizer`, `rank_categorical`, `truncate_categorical`, `capper`, `floorer`, `standard_scaler`, `quantile_biner`, `imputer`, `placeholder_imputer`, `custom_transformer`, `null_injector`, `missing_warner`, `value_mapper`, `selector`, `ecdfer`, `prediction_ranger`. See [references/api.md](references/api.md) for signatures.

## Key Conventions

1. **Currying**: all learners/evaluators/splitters use `@curry` — partially apply all params except the DataFrame
2. **Encoded features**: `onehot_categorizer` creates columns named `fklearn_feat__col==val`; learners auto-detect these via `encode_extra_cols=True` (default)
3. **SHAP**: tree-based learners support `predict_fn(df, apply_shap=True)` for SHAP values
4. **Pipeline kwargs**: `build_pipeline` predict functions accept `**kwargs` and selectively route each kwarg to inner predict functions whose signature includes that parameter (e.g., `apply_shap=True` is routed only to tree learners)
5. **Multiclass**: classification learners add `prediction_0`, `prediction_1`, etc. columns for multiclass objectives

## Production Patterns

- **Serialization**: fklearn has no built-in save/load. Serialize `predict_fn` with `pickle` or `cloudpickle`. The pipeline log's `"__fkml__"` key stores metadata (pipeline structure, features, output columns).
- **Score standardization**: Use `ecdfer(prediction_column="prediction")` in the pipeline to map predictions to a 0–1000 integer score.
- **Calibration**: Use `isotonic_calibration_learner(target_column=TARGET)` after the main learner for probability calibration.
- **Prediction clipping**: Use `prediction_ranger(prediction_min=0, prediction_max=1)` to bound outputs.
- **Predict functions are pure**: After training, `predict_fn` is stateless — safe for concurrent production use.

## Common Gotchas

- `space_time_split_dataset` returns a **function** — call it on df to get 4 DataFrames
- `build_pipeline` requires each learner to have **exactly one unfilled arg** (the df)
- Feature importance dict keys may include `fklearn_feat__` prefixed columns from one-hot encoding
- `validator` returns nested logs — flatten with `extract(results["validator_log"], extractor_fn)` from `fklearn.metrics.pd_extractors` where `extractor_fn` is e.g. `evaluator_extractor("roc_auc")`
- `catboost_regressor_learner` does **not** have `encode_extra_cols` parameter (unlike other learners) — pass cat_features via `extra_params` instead

## Reference

For complete API documentation with all function signatures, see [references/api.md](references/api.md).
