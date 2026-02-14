#!/usr/bin/env python3
"""
Cross-Validation Validator for Tabular ML
Supports multiple CV strategies with proper OOF prediction generation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, 
    StratifiedGroupKFold, TimeSeriesSplit
)
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from typing import Callable, Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CVValidator:
    """Cross-validation with various strategies and proper OOF generation."""
    
    def __init__(
        self, 
        n_splits: int = 5,
        cv_strategy: str = 'kfold',
        shuffle: bool = True,
        random_state: int = 42,
        groups: Optional[np.ndarray] = None,
    ):
        """
        Args:
            n_splits: Number of CV folds
            cv_strategy: One of 'kfold', 'stratified', 'group', 
                        'stratified_group', 'timeseries'
            shuffle: Whether to shuffle before splitting
            random_state: Random seed
            groups: Group labels for group-based CV
        """
        self.n_splits = n_splits
        self.cv_strategy = cv_strategy
        self.shuffle = shuffle
        self.random_state = random_state
        self.groups = groups
        
        self.cv_splitter = self._get_cv_splitter()
        self.fold_scores = []
    
    def _get_cv_splitter(self):
        """Get the appropriate CV splitter."""
        if self.cv_strategy == 'kfold':
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'group':
            return GroupKFold(n_splits=self.n_splits)
        elif self.cv_strategy == 'stratified_group':
            return StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def split(self, X, y=None):
        """Generate CV splits."""
        if self.cv_strategy in ['group', 'stratified_group']:
            if self.groups is None:
                raise ValueError("Groups must be provided for group-based CV")
            return self.cv_splitter.split(X, y, groups=self.groups)
        elif self.cv_strategy == 'stratified':
            return self.cv_splitter.split(X, y)
        else:
            return self.cv_splitter.split(X)
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        fit_params: Optional[Dict[str, Any]] = None,
        predict_proba: bool = False,
        metric_func: Optional[Callable] = None,
        X_test: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[float], float]:
        """
        Run cross-validation with proper OOF prediction generation.
        
        Args:
            X: Training features
            y: Training target
            model_factory: Function that returns a new model instance
            fit_params: Additional parameters for model.fit()
            predict_proba: Use predict_proba instead of predict
            metric_func: Function(y_true, y_pred) -> score
            X_test: Optional test features for predictions
            verbose: Print progress
        
        Returns:
            oof_predictions: Out-of-fold predictions
            test_predictions: Test predictions (averaged across folds)
            fold_scores: List of per-fold scores
            cv_score: Overall CV score
        """
        fit_params = fit_params or {}
        
        if metric_func is None:
            metric_func = lambda y, p: np.sqrt(mean_squared_error(y, p))
        
        oof_predictions = np.zeros(len(X))
        test_predictions = np.zeros(len(X_test)) if X_test is not None else None
        self.fold_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(self.split(X, y)):
            if verbose:
                print(f"\nFold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_factory()
            
            # Handle eval_set for tree models
            if 'eval_set' in fit_params:
                fit_params_fold = fit_params.copy()
                fit_params_fold['eval_set'] = [(X_val, y_val)]
            else:
                fit_params_fold = fit_params
            
            model.fit(X_train, y_train, **fit_params_fold)
            models.append(model)
            
            # Generate predictions
            if predict_proba:
                val_pred = model.predict_proba(X_val)[:, 1]
                if X_test is not None:
                    test_predictions += model.predict_proba(X_test)[:, 1] / self.n_splits
            else:
                val_pred = model.predict(X_val)
                if X_test is not None:
                    test_predictions += model.predict(X_test) / self.n_splits
            
            oof_predictions[val_idx] = val_pred
            
            # Calculate fold score
            fold_score = metric_func(y_val, val_pred)
            self.fold_scores.append(fold_score)
            
            if verbose:
                print(f"Fold {fold + 1} Score: {fold_score:.6f}")
        
        # Overall CV score
        cv_score = metric_func(y, oof_predictions)
        
        if verbose:
            print(f"\n{'='*40}")
            print(f"CV Score: {cv_score:.6f}")
            print(f"Fold Std: {np.std(self.fold_scores):.6f}")
            print(f"{'='*40}")
        
        return oof_predictions, test_predictions, self.fold_scores, cv_score


class TimeBasedValidator:
    """Custom time-based validation with gap period."""
    
    def __init__(
        self,
        n_splits: int = 5,
        gap_days: int = 0,
        test_days: int = 30,
    ):
        """
        Args:
            n_splits: Number of validation folds
            gap_days: Number of days between train end and val start
            test_days: Number of days in each validation period
        """
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.test_days = test_days
    
    def split(self, df: pd.DataFrame, date_column: str):
        """
        Generate time-based splits.
        
        Args:
            df: DataFrame with date column
            date_column: Name of the datetime column
        """
        dates = pd.to_datetime(df[date_column])
        unique_dates = dates.sort_values().unique()
        
        n_dates = len(unique_dates)
        test_size = max(1, n_dates // (self.n_splits + 1))
        
        for i in range(self.n_splits):
            # Train: all dates up to split point
            train_end_idx = n_dates - (self.n_splits - i) * test_size - self.gap_days
            train_end_date = unique_dates[train_end_idx]
            
            # Val: dates after gap
            val_start_idx = train_end_idx + self.gap_days
            val_end_idx = min(val_start_idx + test_size, n_dates - 1)
            
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx]
            
            train_mask = dates <= train_end_date
            val_mask = (dates >= val_start_date) & (dates <= val_end_date)
            
            train_idx = df.index[train_mask].values
            val_idx = df.index[val_mask].values
            
            yield train_idx, val_idx


def check_cv_test_correlation(
    cv_scores: List[float],
    test_scores: List[float],
    verbose: bool = True
) -> float:
    """
    Check if CV scores correlate with test scores.
    
    Args:
        cv_scores: List of CV scores from different experiments
        test_scores: List of corresponding test scores
        verbose: Print analysis
    
    Returns:
        correlation: Pearson correlation coefficient
    """
    if len(cv_scores) < 3:
        print("Need at least 3 experiments to check correlation")
        return np.nan
    
    correlation = np.corrcoef(cv_scores, test_scores)[0, 1]
    
    if verbose:
        if correlation > 0.8:
            assessment = "STRONG - CV is reliable"
        elif correlation > 0.5:
            assessment = "MODERATE - CV somewhat reliable"
        else:
            assessment = "WEAK - Reconsider CV strategy"
        
        print(f"\nCV-Test Correlation: {correlation:.3f}")
        print(f"Assessment: {assessment}")
    
    return correlation


def detect_overfitting(
    train_scores: List[float],
    val_scores: List[float],
    threshold: float = 0.1
) -> bool:
    """
    Detect potential overfitting from train/val score gap.
    
    Args:
        train_scores: Training scores per fold
        val_scores: Validation scores per fold
        threshold: Gap threshold for overfitting warning
    
    Returns:
        is_overfitting: Whether overfitting is detected
    """
    gaps = np.array(train_scores) - np.array(val_scores)
    mean_gap = np.mean(gaps)
    
    is_overfitting = mean_gap > threshold
    
    print(f"\nTrain-Val Gap: {mean_gap:.4f}")
    if is_overfitting:
        print("WARNING: Potential overfitting detected!")
    else:
        print("No significant overfitting detected.")
    
    return is_overfitting


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    import xgboost as xgb
    
    print("Generating sample data...")
    X, y = make_classification(n_samples=10000, n_features=50, n_informative=20,
                                random_state=42)
    X_test = X[:1000]
    
    # Define model factory
    def model_factory():
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            device='cuda',
            tree_method='hist',
            random_state=42,
            early_stopping_rounds=20
        )
    
    # Run CV with stratified folds
    validator = CVValidator(
        n_splits=5,
        cv_strategy='stratified',
        shuffle=True,
        random_state=42
    )
    
    oof, test_pred, fold_scores, cv_score = validator.validate(
        X, y,
        model_factory=model_factory,
        fit_params={'eval_set': True},  # Will be replaced per fold
        predict_proba=True,
        metric_func=roc_auc_score,
        X_test=X_test,
        verbose=True
    )
    
    print(f"\nOOF predictions shape: {oof.shape}")
    print(f"Test predictions shape: {test_pred.shape}")
