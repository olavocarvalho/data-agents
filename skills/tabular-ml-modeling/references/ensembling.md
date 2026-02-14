# Ensembling Techniques

## Why Ensemble?

Different models capture complementary patterns. Combining them:
- Reduces variance and overfitting
- Captures diverse data patterns
- Often yields 1-5% improvement over best single model
- Competition-winning solutions almost always use ensembles

## Hill Climbing Ensemble

Start with the best single model, systematically add others with optimized weights.

### GPU-Accelerated Hill Climbing

```python
import cupy as cp
import numpy as np
from tqdm import tqdm

def rmse_cupy(actual, predicted):
    """GPU-accelerated RMSE calculation."""
    return cp.sqrt(cp.mean((actual - predicted) ** 2))

def roc_auc_cupy(actual, predicted):
    """GPU-accelerated ROC-AUC approximation."""
    n_pos = cp.sum(actual)
    n_neg = len(actual) - n_pos
    ranked = cp.argsort(cp.argsort(predicted)) + 1
    return (cp.sum(ranked[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def hill_climb_ensemble(oof_preds_list, y_true, metric='rmse', n_iterations=1000):
    """
    Hill climbing to find optimal ensemble weights.
    
    Args:
        oof_preds_list: List of OOF predictions from different models
        y_true: True target values
        metric: 'rmse' or 'auc'
        n_iterations: Number of weight combinations to try
    
    Returns:
        best_weights: Optimal weights for each model
        best_score: Best achieved score
    """
    # Move to GPU
    preds = cp.array([cp.array(p) for p in oof_preds_list])
    y = cp.array(y_true)
    n_models = len(oof_preds_list)
    
    # Choose metric function
    if metric == 'rmse':
        score_func = lambda actual, pred: -rmse_cupy(actual, pred)  # Negative for maximization
    else:
        score_func = roc_auc_cupy
    
    # Start with best single model
    single_scores = []
    for i in range(n_models):
        score = score_func(y, preds[i])
        single_scores.append(float(score.get()))
    
    best_idx = np.argmax(single_scores)
    best_weights = np.zeros(n_models)
    best_weights[best_idx] = 1.0
    best_score = single_scores[best_idx]
    
    print(f"Best single model: {best_idx}, Score: {-best_score if metric == 'rmse' else best_score:.6f}")
    
    # Hill climbing iterations
    for iteration in tqdm(range(n_iterations)):
        # Try random perturbation
        new_weights = best_weights.copy()
        
        # Randomly adjust weights
        idx1, idx2 = np.random.choice(n_models, 2, replace=False)
        delta = np.random.uniform(0.01, 0.1)
        
        new_weights[idx1] = max(0, new_weights[idx1] - delta)
        new_weights[idx2] = new_weights[idx2] + delta
        
        # Normalize to sum to 1
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        
        # Calculate blended predictions
        weights_gpu = cp.array(new_weights)
        blended = cp.sum(preds * weights_gpu[:, None], axis=0)
        
        # Score
        new_score = float(score_func(y, blended).get())
        
        if new_score > best_score:
            best_score = new_score
            best_weights = new_weights
            if iteration % 100 == 0:
                score_display = -best_score if metric == 'rmse' else best_score
                print(f"Iteration {iteration}: New best score: {score_display:.6f}")
    
    final_score = -best_score if metric == 'rmse' else best_score
    print(f"\nFinal ensemble score: {final_score:.6f}")
    print(f"Weights: {dict(enumerate(best_weights.round(4)))}")
    
    return best_weights, final_score

# Usage
oof_preds = [xgb_oof, lgb_oof, cat_oof, nn_oof, rf_oof]
weights, score = hill_climb_ensemble(oof_preds, y_train, metric='rmse')

# Apply weights to test predictions
test_preds = [xgb_test, lgb_test, cat_test, nn_test, rf_test]
final_test = np.average(test_preds, weights=weights, axis=0)
```

### Grid Search Variation

```python
def grid_search_ensemble(oof_preds_list, y_true, metric='rmse', step=0.1):
    """Exhaustive grid search for small number of models."""
    from itertools import product
    
    n_models = len(oof_preds_list)
    preds = cp.array([cp.array(p) for p in oof_preds_list])
    y = cp.array(y_true)
    
    score_func = lambda actual, pred: -rmse_cupy(actual, pred) if metric == 'rmse' else roc_auc_cupy(actual, pred)
    
    # Generate all weight combinations
    weight_range = np.arange(0, 1 + step, step)
    best_score = -np.inf
    best_weights = None
    
    for weights in product(weight_range, repeat=n_models):
        weights = np.array(weights)
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()
        
        weights_gpu = cp.array(weights)
        blended = cp.sum(preds * weights_gpu[:, None], axis=0)
        score = float(score_func(y, blended).get())
        
        if score > best_score:
            best_score = score
            best_weights = weights
    
    return best_weights, -best_score if metric == 'rmse' else best_score
```

## Stacking

Train a meta-model on out-of-fold predictions from base models.

### Two-Level Stacking

```python
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

def two_level_stacking(X, y, X_test, base_models, meta_model, n_folds=5):
    """
    Two-level stacking ensemble.
    
    Args:
        X: Training features
        y: Training target
        X_test: Test features
        base_models: List of (name, model) tuples
        meta_model: Model to combine base predictions
        n_folds: Number of CV folds
    
    Returns:
        final_predictions: Stacked predictions for test set
    """
    n_train = len(X)
    n_test = len(X_test)
    n_models = len(base_models)
    
    # Store OOF predictions for stacking
    oof_train = np.zeros((n_train, n_models))
    oof_test = np.zeros((n_test, n_models))
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Train each base model
    for model_idx, (name, model_class) in enumerate(base_models):
        print(f"\nTraining {name}...")
        
        test_preds_folds = np.zeros((n_folds, n_test))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Clone and train model
            model = model_class()
            model.fit(X_tr, y_tr)
            
            # OOF predictions
            oof_train[val_idx, model_idx] = model.predict(X_val)
            test_preds_folds[fold] = model.predict(X_test)
        
        # Average test predictions across folds
        oof_test[:, model_idx] = test_preds_folds.mean(axis=0)
        
        cv_score = np.sqrt(np.mean((y - oof_train[:, model_idx]) ** 2))
        print(f"{name} CV RMSE: {cv_score:.6f}")
    
    # Train meta-model on OOF predictions
    print("\nTraining meta-model...")
    meta_model.fit(oof_train, y)
    
    # Final predictions
    final_predictions = meta_model.predict(oof_test)
    
    return final_predictions, oof_train, oof_test

# Usage
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

base_models = [
    ('xgb', lambda: XGBRegressor(device='cuda', n_estimators=500)),
    ('lgb', lambda: LGBMRegressor(device='gpu', n_estimators=500)),
    ('cat', lambda: CatBoostRegressor(task_type='GPU', iterations=500, verbose=0)),
]

meta_model = Ridge(alpha=1.0)

predictions, oof_train, oof_test = two_level_stacking(
    X_train, y_train, X_test, base_models, meta_model
)
```

### Three-Level Stacking

```python
def three_level_stacking(X, y, X_test, level1_models, level2_models, level3_model, n_folds=5):
    """
    Three-level stacking for maximum performance.
    
    Level 1: Diverse base models
    Level 2: Meta-learners on L1 predictions
    Level 3: Final blender on L2 predictions
    """
    # Level 1: Base models
    l1_train, l1_test = generate_stacking_features(
        X, y, X_test, level1_models, n_folds
    )
    
    # Level 2: Combine L1 with original features
    X_l2_train = np.hstack([l1_train, X])
    X_l2_test = np.hstack([l1_test, X_test])
    
    l2_train, l2_test = generate_stacking_features(
        X_l2_train, y, X_l2_test, level2_models, n_folds
    )
    
    # Level 3: Final prediction
    level3_model.fit(l2_train, y)
    final_predictions = level3_model.predict(l2_test)
    
    return final_predictions

def generate_stacking_features(X, y, X_test, models, n_folds):
    """Generate stacking features from models."""
    n_train = len(X)
    n_test = len(X_test)
    n_models = len(models)
    
    stack_train = np.zeros((n_train, n_models))
    stack_test = np.zeros((n_test, n_models))
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for model_idx, (name, model_factory) in enumerate(models):
        test_preds_folds = []
        
        for train_idx, val_idx in kfold.split(X):
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            
            stack_train[val_idx, model_idx] = model.predict(X[val_idx])
            test_preds_folds.append(model.predict(X_test))
        
        stack_test[:, model_idx] = np.mean(test_preds_folds, axis=0)
    
    return stack_train, stack_test
```

## Pseudo-Labeling

Use model predictions on unlabeled data to expand training set.

### Basic Pseudo-Labeling

```python
def pseudo_label_training(X_train, y_train, X_test, model_factory, 
                          confidence_threshold=0.95, n_iterations=3):
    """
    Iterative pseudo-labeling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Unlabeled test features
        model_factory: Function that returns a new model instance
        confidence_threshold: Only use predictions above this confidence
        n_iterations: Number of pseudo-labeling rounds
    """
    X_pseudo = X_train.copy()
    y_pseudo = y_train.copy()
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Train on current data
        model = model_factory()
        model.fit(X_pseudo, y_pseudo)
        
        # Predict on test (with probabilities for classification)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)
            confidence = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
        else:
            # For regression, use prediction variance or ensemble disagreement
            predictions = model.predict(X_test)
            confidence = np.ones(len(X_test))  # All pseudo-labels for regression
        
        # Select high-confidence samples
        high_conf_mask = confidence >= confidence_threshold
        n_selected = high_conf_mask.sum()
        print(f"Selected {n_selected} high-confidence samples")
        
        if n_selected == 0:
            break
        
        # Add pseudo-labeled samples to training
        X_pseudo = np.vstack([X_pseudo, X_test[high_conf_mask]])
        y_pseudo = np.concatenate([y_pseudo, predictions[high_conf_mask]])
        
        # Remove used test samples
        X_test = X_test[~high_conf_mask]
        
        print(f"New training size: {len(X_pseudo)}")
    
    # Final model on all pseudo-labeled data
    final_model = model_factory()
    final_model.fit(X_pseudo, y_pseudo)
    
    return final_model

# Usage
model = pseudo_label_training(
    X_train, y_train, X_test,
    model_factory=lambda: XGBRegressor(device='cuda', n_estimators=500),
    confidence_threshold=0.9,
    n_iterations=3
)
```

### Soft Pseudo-Labels

```python
def soft_pseudo_label_training(X_train, y_train, X_test, base_model, 
                                weight_factor=0.5):
    """
    Use soft pseudo-labels (probabilities) instead of hard labels.
    More robust than hard pseudo-labeling.
    """
    # Train initial model
    base_model.fit(X_train, y_train)
    
    # Get soft predictions for test
    if hasattr(base_model, 'predict_proba'):
        soft_labels = base_model.predict_proba(X_test)
    else:
        soft_labels = base_model.predict(X_test)
    
    # Combine original and pseudo-labeled data
    X_combined = np.vstack([X_train, X_test])
    
    # Sample weights: original data = 1.0, pseudo-labeled = weight_factor
    sample_weights = np.concatenate([
        np.ones(len(X_train)),
        np.full(len(X_test), weight_factor)
    ])
    
    # For classification with soft labels, use custom training loop
    # For regression, concatenate targets
    if soft_labels.ndim == 1:
        y_combined = np.concatenate([y_train, soft_labels])
    
    # Train final model with sample weights
    final_model = XGBRegressor(device='cuda', n_estimators=500)
    final_model.fit(X_combined, y_combined, sample_weight=sample_weights)
    
    return final_model
```

## Seed Averaging

Train same model with different random seeds, average predictions.

```python
def seed_averaging(X_train, y_train, X_test, model_factory, n_seeds=10):
    """
    Average predictions from models trained with different seeds.
    Simple but effective technique for reducing variance.
    """
    predictions = []
    oof_preds = []
    
    for seed in tqdm(range(n_seeds)):
        model = model_factory(seed)
        
        # Train on full data
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
    
    # Average predictions
    final_predictions = np.mean(predictions, axis=0)
    
    # Calculate ensemble benefit
    single_std = np.std([p.std() for p in predictions])
    ensemble_std = final_predictions.std()
    print(f"Ensemble reduced prediction variance by {(1 - ensemble_std/single_std)*100:.1f}%")
    
    return final_predictions

# Usage
def model_factory(seed):
    return XGBRegressor(
        device='cuda',
        n_estimators=500,
        random_state=seed
    )

final_predictions = seed_averaging(X_train, y_train, X_test, model_factory, n_seeds=10)
```

## Ensemble Strategy Summary

| Technique | When to Use | Expected Improvement |
|-----------|-------------|---------------------|
| Simple Average | Quick baseline | 1-2% |
| Hill Climbing | Diverse models | 2-4% |
| Stacking | Competition winning | 3-5% |
| Pseudo-Labeling | Abundant unlabeled data | 1-3% |
| Seed Averaging | Reduce variance | 0.5-1% |

### Competition-Winning Combination

```python
# 1. Train diverse base models with CV
models = train_diverse_baselines(X, y, X_test)

# 2. Hill climb to find optimal weights
weights, _ = hill_climb_ensemble(models['oof'], y)
blend1 = weighted_average(models['test'], weights)

# 3. Stack with meta-model
stack_pred, _, _ = two_level_stacking(X, y, X_test, base_models, Ridge())

# 4. Pseudo-label high confidence predictions
pseudo_model = pseudo_label_training(X, y, X_test, XGBRegressor)
pseudo_pred = pseudo_model.predict(X_test)

# 5. Final blend
final = 0.4 * blend1 + 0.4 * stack_pred + 0.2 * pseudo_pred
```
