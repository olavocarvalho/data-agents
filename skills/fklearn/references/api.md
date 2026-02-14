# fklearn API Reference

Complete API reference derived from fklearn source code (github.com/nubank/fklearn).

## Type Definitions (`fklearn.types`)

```python
DateType = Union[pd.Period, datetime, str]
LogType = Dict[str, Any]
PredictFnType = Callable[..., pd.DataFrame]
LearnerLogType = Dict[str, Any]
LearnerReturnType = Tuple[PredictFnType, pd.DataFrame, LearnerLogType]
LearnerFnType = Callable[[pd.DataFrame], LearnerReturnType]
EvalReturnType = Dict[str, Union[float, Dict]]
EvalFnType = Callable[[pd.DataFrame], EvalReturnType]
SplitterReturnType = Tuple[FoldType, LogListType]
```

---

## Preprocessing — Splitting (`fklearn.preprocessing.splitting`)

### space_time_split_dataset

4-way panel data split by ID and time. Returns a **curried function** — call it on df.

```python
split_fn = space_time_split_dataset(
    train_start_date: DateType,        # Start of training period
    train_end_date: DateType,          # End of training / start of holdout
    holdout_end_date: DateType,        # End of holdout period
    split_seed: int,                   # Seed for reproducible ID sampling
    space_holdout_percentage: float,   # Proportion of IDs to hold out (e.g., 0.2)
    space_column: str,                 # ID column name
    time_column: str,                  # Date column name
    holdout_space: np.ndarray = None,  # Pre-defined holdout IDs (optional)
    holdout_start_date: DateType = None,  # Custom holdout start (defaults to train_end_date)
)
train, intime_outspace, outtime_inspace, outtime_outspace = split_fn(df)
```

**Returns 4 DataFrames:**
1. `train` — in-time, in-space (training set)
2. `intime_outspace` — in-time, out-of-space (OOS holdout)
3. `outtime_inspace` — out-of-time, in-space (OOT holdout)
4. `outtime_outspace` — out-of-time, out-of-space (OOST holdout)

### time_split_dataset

Simple temporal split into train/test.

```python
split_fn = time_split_dataset(
    train_start_date: DateType,
    train_end_date: DateType,
    holdout_end_date: DateType,
    time_column: str,
    holdout_start_date: DateType = None,
)
train_set, test_set = split_fn(df)
```

### stratified_split_dataset

Stratified random split preserving class ratios.

```python
split_fn = stratified_split_dataset(
    target_column: str,
    test_size: float,              # Proportion for test (0.0 to 1.0)
    random_state: int = None,
)
train_set, test_set = split_fn(df)
```

---

## Preprocessing — Rebalancing (`fklearn.preprocessing.rebalancing`)

### rebalance_by_categorical

Downsample to equal category sizes.

```python
rebalance_fn = rebalance_by_categorical(
    categ_column: str,
    max_lines_by_categ: int = None,  # Max rows per category (None = min category size)
    seed: int = None,
)
balanced_df = rebalance_fn(df)
```

### rebalance_by_continuous

Bucket a continuous column then balance.

```python
rebalance_fn = rebalance_by_continuous(
    continuous_column: str,
    buckets: int = 10,
    by_quantile: bool = True,
    max_lines_by_categ: int = None,
    seed: int = None,
)
```

---

## Training — Classification (`fklearn.training.classification`)

All classification learners return `LearnerReturnType = (predict_fn, predicted_df, log)`.

### lgbm_classification_learner

LightGBM classifier for binary and multi-class classification.

```python
learner = lgbm_classification_learner(
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Dict = None,           # e.g., {"num_leaves": 31, "max_depth": -1, "seed": 42, "verbose": -1}
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,      # Auto-detect fklearn_feat__ columns
    valid_sets: List[pd.DataFrame] = None,  # For early stopping
    valid_names: List[str] = None,
    feval: Callable = None,              # Custom eval function
    init_model: Union[str, Path, Booster] = None,  # Continue training
    feature_name: Union[List[str], 'auto'] = 'auto',
    categorical_feature: Union[List, 'auto'] = 'auto',
    keep_training_booster: bool = False,
    callbacks: List[Callable] = None,
    dataset_init_score = None,
)
predict_fn, predicted_df, log = learner(train_df)
# predict_fn supports: predict_fn(df, apply_shap=True) for SHAP values
```

**Log keys:** `log["lgbm_classification_learner"]` contains `features`, `target`, `prediction_column`, `package`, `package_version`, `parameters`, `feature_importance` (dict), `training_samples`.

### xgb_classification_learner

XGBoost classifier.

```python
learner = xgb_classification_learner(
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Dict = None,           # e.g., {"max_depth": 6, "objective": "binary:logistic"}
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
# Supports multiclass with extra_params={"objective": "multi:softprob", "num_class": N}
# predict_fn supports apply_shap=True
```

### catboost_classification_learner

CatBoost classifier with native categorical feature support.

```python
learner = catboost_classification_learner(
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Dict = None,           # e.g., {"cat_features": [0, 1]}
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
# Multiclass with extra_params={"objective": "MultiClass"}
```

### logistic_classification_learner

Scikit-learn LogisticRegression.

```python
learner = logistic_classification_learner(
    features: List[str],
    target: str,
    params: Dict = None,                 # Default: {"C": 0.1, "multi_class": "ovr", "solver": "liblinear"}
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
```

### nlp_logistic_classification_learner

TF-IDF vectorizer + logistic regression for text classification.

```python
learner = nlp_logistic_classification_learner(
    text_feature_cols: List[str],        # Text columns (concatenated)
    target: str,
    vectorizer_params: Dict = None,      # TfidfVectorizer params
    logistic_params: Dict = None,        # LogisticRegression params
    prediction_column: str = "prediction",
)
```

---

## Training — Regression (`fklearn.training.regression`)

### lgbm_regression_learner

```python
learner = lgbm_regression_learner(
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Dict = None,
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
# Supports apply_shap=True
```

### xgb_regression_learner

```python
learner = xgb_regression_learner(
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Dict = None,           # Default objective: "reg:linear"
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
```

### catboost_regressor_learner

```python
learner = catboost_regressor_learner(
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Dict = None,
    prediction_column: str = "prediction",
    weight_column: str = None,
)
```

### linear_regression_learner

```python
learner = linear_regression_learner(
    features: List[str],
    target: str,
    params: Dict = None,                 # Default: {"fit_intercept": True}
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
```

### elasticnet_regression_learner

```python
learner = elasticnet_regression_learner(
    features: List[str],
    target: str,
    params: Dict = None,
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
```

### gp_regression_learner

Gaussian Process regressor.

```python
learner = gp_regression_learner(
    features: List[str],
    target: str,
    kernel: sklearn.gaussian_process.kernels.Kernel = None,
    alpha: float = 0.1,
    extra_variance: Union[str, float] = "fit",  # "fit" uses target std
    return_std: bool = False,            # If True, adds prediction_std column
    extra_params: Dict = None,
    prediction_column: str = "prediction",
    encode_extra_cols: bool = True,
)
```

### custom_supervised_model_learner

Wraps any sklearn-compatible model.

```python
learner = custom_supervised_model_learner(
    features: List[str],
    target: str,
    model: Any,                          # Must have .fit(); .predict_proba() for classification, .predict() for regression
    supervised_type: str,                # "classification" or "regression"
    log: Dict[str, Dict],               # Must have exactly 1 key (model name)
    prediction_column: str = "prediction",
)
```

---

## Training — Transformation (`fklearn.training.transformation`)

All transformations are curried learners returning `(predict_fn, transformed_df, log)`.

### label_categorizer

Encode categorical columns as integer labels.

```python
fn = label_categorizer(
    columns_to_categorize: List[str],
    replace_unseen: Union[str, float] = nan,  # Value for unseen categories at predict time
    store_mapping: bool = False,
)
```

### onehot_categorizer

One-hot encode categorical columns. Creates `fklearn_feat__col==val` columns and drops originals.

```python
fn = onehot_categorizer(
    columns_to_categorize: List[str],
    hardcode_nans: bool = False,         # Extra column for NaN/unseen values
    drop_first_column: bool = False,     # K-1 encoding to avoid colinearity
    store_mapping: bool = False,
)
```

### target_categorizer

Encode categoricals with smoothed target mean (target encoding).

```python
fn = target_categorizer(
    columns_to_categorize: List[str],
    target_column: str,
    smoothing: float = 1.0,             # Weight for global mean vs category mean
    ignore_unseen: bool = True,
    store_mapping: bool = False,
)
```

### count_categorizer

Replace categorical values with their frequency counts.

```python
fn = count_categorizer(
    columns_to_categorize: List[str],
    replace_unseen: int = -1,
    store_mapping: bool = False,
)
```

### rank_categorical

Replace categorical values with frequency rank.

```python
fn = rank_categorical(
    columns_to_rank: List[str],
    replace_unseen: Union[str, float] = nan,
    store_mapping: bool = False,
)
```

### truncate_categorical

Replace infrequent categories with a single value.

```python
fn = truncate_categorical(
    columns_to_truncate: List[str],
    percentile: float,                   # Categories below this frequency % are replaced
    replacement: Union[str, float] = -9999,
    replace_unseen: Union[str, float] = -9999,
    store_mapping: bool = False,
)
```

### capper

Cap column values at maximum (learned from training data or precomputed).

```python
fn = capper(
    columns_to_cap: List[str],
    precomputed_caps: Dict[str, float] = None,  # e.g., {"income": 50000}
)
```

### floorer

Floor column values at minimum.

```python
fn = floorer(
    columns_to_floor: List[str],
    precomputed_floors: Dict[str, float] = None,
)
```

### prediction_ranger

Clip prediction column to a range.

```python
fn = prediction_ranger(
    prediction_min: float,
    prediction_max: float,
    prediction_column: str = "prediction",
)
```

### ecdfer

Empirical CDF — convert predictions to 0-1000 score.

```python
fn = ecdfer(
    ascending: bool = True,
    prediction_column: str = "prediction",
    ecdf_column: str = "prediction_ecdf",
    max_range: int = 1000,
)
```

### discrete_ecdfer

Discretized ECDF with configurable rounding.

```python
fn = discrete_ecdfer(
    ascending: bool = True,
    prediction_column: str = "prediction",
    ecdf_column: str = "prediction_ecdf",
    max_range: int = 1000,
    round_method: Callable = int,        # int, ceil, floor, round
)
```

### standard_scaler

Z-score standardization.

```python
fn = standard_scaler(columns_to_scale: List[str])
```

### quantile_biner

Discretize continuous columns into quantile bins.

```python
fn = quantile_biner(
    columns_to_bin: List[str],
    q: int = 4,                          # Number of quantiles
    right: bool = False,
)
```

### value_mapper

Map values according to dictionaries.

```python
fn = value_mapper(
    value_maps: Dict[str, Dict],         # {"col": {old_val: new_val, ...}}
    ignore_unseen: bool = True,
    replace_unseen_to: Any = nan,
)
```

### selector

Select specific columns from DataFrame.

```python
fn = selector(
    training_columns: List[str],
    predict_columns: List[str] = None,   # Defaults to training_columns
)
```

### custom_transformer

Apply a custom function to columns.

```python
fn = custom_transformer(
    columns_to_transform: List[str],
    transformation_function: Callable,
    is_vectorized: bool = False,         # True for vectorized ops, False uses swifter.apply
)
```

### null_injector

Inject NaN values for robustness testing.

```python
fn = null_injector(
    proportion: float,                   # 0.0 to 1.0
    columns_to_inject: List[str] = None,
    groups: List[List[str]] = None,      # Group features to null together
    seed: int = 1,
)
```

### missing_warner

Flag rows with unexpected missing values at predict time.

```python
fn = missing_warner(
    cols_list: List[str],
    new_column_name: str = "has_unexpected_missing",
    detailed_warning: bool = False,
    detailed_column_name: str = None,
)
```

---

## Training — Imputation (`fklearn.training.imputation`)

### imputer

Impute missing values using sklearn strategies.

```python
fn = imputer(
    columns_to_impute: List[str],
    impute_strategy: str = "median",     # "median", "mean", "most_frequent"
    placeholder_value: float = None,
)
```

### placeholder_imputer

Fill NaN with a fixed value.

```python
fn = placeholder_imputer(
    columns_to_impute: List[str],
    placeholder_value: float = -999,
)
```

---

## Training — Calibration (`fklearn.training.calibration`)

### isotonic_calibration_learner

Isotonic regression for probability calibration.

```python
fn = isotonic_calibration_learner(
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_column: str = "calibrated_prediction",
    y_min: float = 0.0,
    y_max: float = 1.0,
)
```

---

## Training — Pipeline (`fklearn.training.pipeline`)

### build_pipeline

Compose multiple curried learners into a single learner function.

```python
pipeline = build_pipeline(
    *learners: LearnerFnType,           # Curried learners (must have exactly 1 unfilled arg)
    has_repeated_learners: bool = False, # Set True if pipeline contains duplicate learner names
)
predict_fn, transformed_df, logs = pipeline(train_df)

# Pipeline predict functions pass **kwargs to inner predict fns:
test_preds = predict_fn(test_df, apply_shap=True)
```

**Logs structure:**
- Merged dict of all individual learner logs
- `logs["__fkml__"]` contains: `pipeline` (list of learner names), `output_columns`, `features`, `learners` (serialization data)

---

## Training — Unsupervised (`fklearn.training.unsupervised`)

### isolation_forest_learner

Anomaly detection with Isolation Forest.

```python
fn = isolation_forest_learner(
    features: List[str],
    params: Dict = None,                 # sklearn IsolationForest params
    prediction_column: str = "prediction",
    encode_extra_cols: bool = True,
)
```

---

## Training — Ensemble (`fklearn.training.ensemble`)

### xgb_octopus_classification_learner

Domain-knowledge ensemble: trains separate XGBoost models per category.

```python
fn = xgb_octopus_classification_learner(
    train_split_col: str,                # Column to split by
    train_split_bins: List,              # Category values for each bin
    features_by_bin: List[List[str]],    # Features per bin
    learning_rate_by_bin: List[float],
    num_estimators_by_bin: List[int],
    extra_params_by_bin: List[Dict] = None,
    prediction_column: str = "prediction",
    weight_column: str = None,
    encode_extra_cols: bool = True,
)
```

---

## Validation — Evaluators (`fklearn.validation.evaluators`)

All evaluators are curried. Common parameters: `prediction_column="prediction"`, `target_column="target"`, `weight_column=None`, `eval_name=None`.

### Classification Evaluators

```python
# ROC AUC
eval_fn = roc_auc_evaluator(prediction_column="prediction", target_column="target", eval_name="roc_auc")

# Precision-Recall AUC
eval_fn = pr_auc_evaluator(prediction_column="prediction", target_column="target", eval_name="pr_auc")

# Precision (with threshold)
eval_fn = precision_evaluator(threshold=0.5, prediction_column="prediction", target_column="target")

# Recall (with threshold)
eval_fn = recall_evaluator(threshold=0.5, prediction_column="prediction", target_column="target")

# F-beta score
eval_fn = fbeta_score_evaluator(threshold=0.5, beta=1.0, prediction_column="prediction", target_column="target")

# Log loss
eval_fn = logloss_evaluator(prediction_column="prediction", target_column="target")

# Brier score
eval_fn = brier_score_evaluator(prediction_column="prediction", target_column="target")

# Expected Calibration Error
eval_fn = expected_calibration_error_evaluator(
    prediction_column="prediction", target_column="target",
    n_bins=100, bin_choice="count"  # "count" for equally populated, "prob" for equally spaced
)
```

### Regression Evaluators

```python
# R-squared
eval_fn = r2_evaluator(prediction_column="prediction", target_column="target")

# Mean Squared Error
eval_fn = mse_evaluator(prediction_column="prediction", target_column="target")

# Mean prediction value
eval_fn = mean_prediction_evaluator(prediction_column="prediction")
```

### Correlation Evaluators

```python
eval_fn = correlation_evaluator(...)      # Pearson correlation
eval_fn = spearman_evaluator(...)         # Spearman rank correlation
eval_fn = linear_coefficient_evaluator(...)    # Linear regression coefficient
eval_fn = exponential_coefficient_evaluator(...)  # Exponential model coefficient
eval_fn = logistic_coefficient_evaluator(...)     # Logistic model coefficient
```

### Ranking Evaluator

```python
eval_fn = ndcg_evaluator(
    prediction_column="prediction", target_column="target",
    k: int = None,                       # Top-k (None = all)
    exponential_gain: bool = True,       # False for linear gain
)
```

### Meta-Evaluators

```python
# Combine multiple evaluators
eval_fn = combined_evaluators(evaluators=[
    roc_auc_evaluator(target_column="target", eval_name="roc_auc"),
    pr_auc_evaluator(target_column="target", eval_name="pr_auc"),
])

# Evaluate by categorical split
eval_fn = split_evaluator(
    eval_fn=roc_auc_evaluator(target_column="target"),
    split_col="region",
    split_values=None,                   # None = all unique values
)

# Evaluate by temporal split
eval_fn = temporal_split_evaluator(
    eval_fn=roc_auc_evaluator(target_column="target"),
    time_col="date",
    time_format="%Y-%m",
)

# Permutation importance
eval_fn = permutation_evaluator(
    predict_fn=predict_fn,
    eval_fn=roc_auc_evaluator(target_column="target"),
    baseline=True,
    features=None,                       # None = all columns
    shuffle_all_at_once=False,
    random_state=None,
)

# Hash evaluator (reproducibility check)
eval_fn = hash_evaluator(hash_columns=None, consider_index=False)
```

---

## Validation — Splitters (`fklearn.validation.splitters`)

All splitters return `(folds, logs)` where folds = `List[Tuple[train_idx, List[test_idx]]]`.

### k_fold_splitter

Standard K-Fold with optional stratification.

```python
split_fn = k_fold_splitter(
    n_splits: int,
    random_state: int = None,
    stratify_column: str = None,         # Column for stratified splits
)
```

### out_of_time_and_space_splitter

K-Fold by ID with temporal holdout.

```python
split_fn = out_of_time_and_space_splitter(
    n_splits: int,
    in_time_limit: DateType,
    time_column: str,
    space_column: str,
    holdout_gap: timedelta = timedelta(days=0),
)
```

### stability_curve_time_splitter

Fixed training, rolling test window. Shows performance degradation over time.

```python
split_fn = stability_curve_time_splitter(
    training_time_limit: DateType,
    time_column: str,
    freq: str = 'M',                     # Pandas offset alias (D, W, M, etc.)
    min_samples: int = 1000,
)
```

### stability_curve_time_in_space_splitter

Stability curve where all holdout IDs appear in training.

```python
split_fn = stability_curve_time_in_space_splitter(
    training_time_limit: DateType,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    random_state: int = None,
    min_samples: int = 1000,
)
```

### stability_curve_time_space_splitter

Stability curve where no holdout IDs appear in training.

```python
split_fn = stability_curve_time_space_splitter(
    training_time_limit: DateType,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    random_state: int = None,
    min_samples: int = 1000,
)
```

### time_learning_curve_splitter

Progressively larger training sets (temporal). Shows learning curve.

```python
split_fn = time_learning_curve_splitter(
    training_time_limit: DateType,
    time_column: str,
    freq: str = 'M',
    holdout_gap: timedelta = timedelta(days=0),
    min_samples: int = 1000,
)
```

### time_and_space_learning_curve_splitter

Learning curve with out-of-ID holdout.

```python
split_fn = time_and_space_learning_curve_splitter(
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    holdout_gap: timedelta = timedelta(days=0),
    random_state: int = None,
    min_samples: int = 1000,
)
```

### reverse_time_learning_curve_splitter

Inverse learning curve — adds older data progressively.

```python
split_fn = reverse_time_learning_curve_splitter(
    time_column: str,
    training_time_limit: DateType,
    lower_time_limit: DateType = None,
    freq: str = 'MS',
    holdout_gap: timedelta = timedelta(days=0),
    min_samples: int = 1000,
)
```

### spatial_learning_curve_splitter

Learning curve by progressively adding more IDs.

```python
split_fn = spatial_learning_curve_splitter(
    space_column: str,
    time_column: str,
    training_limit: DateType,
    holdout_gap: timedelta = timedelta(days=0),
    train_percentages: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    random_state: int = None,
)
```

### forward_stability_curve_time_splitter

Both training and test windows slide forward. Simulates periodic retraining.

```python
split_fn = forward_stability_curve_time_splitter(
    training_time_start: DateType,
    training_time_end: DateType,
    time_column: str,
    holdout_gap: timedelta = timedelta(days=0),
    holdout_size: timedelta = timedelta(days=90),
    step: timedelta = timedelta(days=90),
    move_training_start_with_steps: bool = True,
)
```

---

## Validation — Validator (`fklearn.validation.validator`)

### validator

Cross-validation orchestrator.

```python
results = validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,            # Curried splitter
    train_fn: LearnerFnType,             # Curried learner or pipeline
    eval_fn: EvalFnType,                 # Curried evaluator
    perturb_fn_train: PerturbFnType = identity,
    perturb_fn_test: PerturbFnType = identity,
    predict_oof: bool = False,           # Return out-of-fold predictions
    return_eval_logs_on_train: bool = False,
    return_all_train_logs: bool = False,
    verbose: bool = False,
    drop_empty_folds: bool = False,
)
```

**Returns dict with:**
- `train_log`: log from first fold training
- `validator_log`: list of per-fold results, each with `eval_results`, `split_log`, `fold_num`

### parallel_validator

Multi-process cross-validation.

```python
results = parallel_validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = 1,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
)
```

---

## Metrics — Extractors (`fklearn.metrics.pd_extractors`)

Utilities to convert nested validation logs into DataFrames.

```python
from fklearn.metrics.pd_extractors import extract, evaluator_extractor, combined_evaluator_extractor

# Convert validator results to DataFrame (extract is curried: needs validator_log + extractor)
extractor = evaluator_extractor("roc_auc")
df_results = extract(validator_results["validator_log"], extractor)

# For multiple metrics, use combined_evaluator_extractor
multi_extractor = combined_evaluator_extractor(base_extractors=[
    evaluator_extractor("roc_auc"),
    evaluator_extractor("pr_auc"),
])
df_results = extract(validator_results["validator_log"], multi_extractor)
```

---

## Tuning (`fklearn.tuning`)

### random_search_tuner

```python
from fklearn.tuning.parameter_tuners import random_search_tuner

results = random_search_tuner(
    space: Dict[str, Callable],          # {"param_name": lambda: random_value}
    train_set: pd.DataFrame,
    param_train_fn: Callable,            # Function that takes params dict and returns LearnerFnType
    split_fn: SplitterFnType,
    eval_fn: EvalFnType,
    iterations: int = 10,
    random_seed: int = None,
    save_intermediary_fn: Callable = None,
    n_jobs: int = 1,
)
```

### grid_search_cv

```python
from fklearn.tuning.parameter_tuners import grid_search_cv

results = grid_search_cv(
    space: Dict[str, List],              # {"param_name": [value1, value2, ...]}
    train_set: pd.DataFrame,
    param_train_fn: Callable,
    split_fn: SplitterFnType,
    eval_fn: EvalFnType,
    warm_start_file: str = None,         # Resume from file
)
```

---

## Causal Inference (`fklearn.causal`)

### Effects

```python
from fklearn.causal.effects import linear_effect, spearman_effect, pearson_effect

effect = linear_effect(df, treatment_column="treatment", outcome_column="outcome")
```

### Debiasing (Frisch-Waugh-Lovell)

```python
from fklearn.causal.debias import debias_with_regression

debias_fn = debias_with_regression(
    treatment_column: str,
    outcome_column: str,
    confounder_columns: List[str],
    suffix: str = "_debiased",
    denoise: bool = True,
)
debiased_df = debias_fn(df)
```

### Double Machine Learning

```python
from fklearn.causal.cate_learning.double_machine_learning import non_parametric_double_ml_learner

learner = non_parametric_double_ml_learner(
    feature_columns: List[str],
    treatment_column: str,
    outcome_column: str,
    debias_model=GradientBoostingRegressor(),
    denoise_model=GradientBoostingRegressor(),
    final_model=LinearRegression(),
    cv_splits: int = 3,
)
```

---

## Data (`fklearn.data.datasets`)

```python
from fklearn.data.datasets import make_tutorial_data, make_confounded_data

# Fake data for tutorials (3 numerical + 2 categorical features)
df = make_tutorial_data(n=1000)

# Confounded data for causal inference studies
df = make_confounded_data(n=1000)
```
