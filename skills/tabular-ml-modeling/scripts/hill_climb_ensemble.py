#!/usr/bin/env python3
"""
Hill Climbing Ensemble Optimizer
GPU-accelerated ensemble weight optimization using CuPy.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available, falling back to NumPy")


class HillClimbingEnsemble:
    """GPU-accelerated hill climbing for ensemble weight optimization."""
    
    def __init__(
        self,
        metric: str = 'rmse',
        n_iterations: int = 1000,
        random_state: int = 42,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            metric: 'rmse', 'mae', 'auc', 'logloss'
            n_iterations: Number of hill climbing iterations
            random_state: Random seed
            use_gpu: Whether to use GPU (CuPy)
            verbose: Print progress
        """
        self.metric = metric
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = verbose
        
        self.best_weights = None
        self.best_score = None
        self.xp = cp if self.use_gpu else np
        
        np.random.seed(random_state)
        if self.use_gpu:
            cp.random.seed(random_state)
    
    def _to_device(self, arr):
        """Move array to GPU if using CuPy."""
        if self.use_gpu and not isinstance(arr, cp.ndarray):
            return cp.array(arr)
        return arr
    
    def _to_host(self, arr):
        """Move array to CPU."""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return arr.get()
        return arr
    
    def _rmse(self, y_true, y_pred):
        """GPU-accelerated RMSE."""
        return self.xp.sqrt(self.xp.mean((y_true - y_pred) ** 2))
    
    def _mae(self, y_true, y_pred):
        """GPU-accelerated MAE."""
        return self.xp.mean(self.xp.abs(y_true - y_pred))
    
    def _auc(self, y_true, y_pred):
        """GPU-accelerated ROC-AUC approximation."""
        n_pos = self.xp.sum(y_true)
        n_neg = len(y_true) - n_pos
        ranked = self.xp.argsort(self.xp.argsort(y_pred)) + 1
        return (self.xp.sum(ranked[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    def _logloss(self, y_true, y_pred):
        """GPU-accelerated log loss."""
        eps = 1e-15
        y_pred = self.xp.clip(y_pred, eps, 1 - eps)
        return -self.xp.mean(y_true * self.xp.log(y_pred) + 
                            (1 - y_true) * self.xp.log(1 - y_pred))
    
    def _get_score_func(self):
        """Get the scoring function."""
        funcs = {
            'rmse': lambda y, p: -self._rmse(y, p),  # Negative for maximization
            'mae': lambda y, p: -self._mae(y, p),
            'auc': self._auc,
            'logloss': lambda y, p: -self._logloss(y, p),
        }
        return funcs.get(self.metric)
    
    def _blend_predictions(self, preds, weights):
        """Create weighted blend of predictions."""
        weights_arr = self.xp.array(weights)
        return self.xp.sum(preds * weights_arr[:, None], axis=0)
    
    def fit(
        self,
        predictions: List[np.ndarray],
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Find optimal ensemble weights via hill climbing.
        
        Args:
            predictions: List of OOF predictions from different models
            y_true: True target values
        
        Returns:
            best_weights: Optimal weights for each model
            best_score: Best achieved score
        """
        # Move to GPU
        preds = self.xp.array([self._to_device(p) for p in predictions])
        y = self._to_device(y_true)
        n_models = len(predictions)
        
        score_func = self._get_score_func()
        
        # Find best single model
        single_scores = []
        for i in range(n_models):
            score = float(self._to_host(score_func(y, preds[i])))
            single_scores.append(score)
        
        best_idx = np.argmax(single_scores)
        self.best_weights = np.zeros(n_models)
        self.best_weights[best_idx] = 1.0
        self.best_score = single_scores[best_idx]
        
        if self.verbose:
            print(f"Best single model: {best_idx}")
            display_score = -self.best_score if self.metric in ['rmse', 'mae', 'logloss'] else self.best_score
            print(f"Single model score: {display_score:.6f}")
        
        # Hill climbing
        iterator = tqdm(range(self.n_iterations)) if self.verbose else range(self.n_iterations)
        
        for iteration in iterator:
            # Random perturbation
            new_weights = self.best_weights.copy()
            
            # Randomly select two models to adjust
            idx1, idx2 = np.random.choice(n_models, 2, replace=False)
            delta = np.random.uniform(0.01, 0.1)
            
            # Shift weight from idx1 to idx2
            new_weights[idx1] = max(0, new_weights[idx1] - delta)
            new_weights[idx2] = new_weights[idx2] + delta
            
            # Normalize
            if new_weights.sum() > 0:
                new_weights = new_weights / new_weights.sum()
            else:
                continue
            
            # Calculate blended prediction
            blended = self._blend_predictions(preds, new_weights)
            new_score = float(self._to_host(score_func(y, blended)))
            
            # Update if improved
            if new_score > self.best_score:
                self.best_score = new_score
                self.best_weights = new_weights
                
                if self.verbose and iteration % 100 == 0:
                    display_score = -self.best_score if self.metric in ['rmse', 'mae', 'logloss'] else self.best_score
                    tqdm.write(f"Iteration {iteration}: New best score: {display_score:.6f}")
        
        final_score = -self.best_score if self.metric in ['rmse', 'mae', 'logloss'] else self.best_score
        
        if self.verbose:
            print(f"\nFinal ensemble score: {final_score:.6f}")
            print(f"Weights: {dict(enumerate(self.best_weights.round(4)))}")
        
        return self.best_weights, final_score
    
    def predict(
        self,
        test_predictions: List[np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply ensemble weights to test predictions.
        
        Args:
            test_predictions: List of test predictions from each model
            weights: Optional custom weights (uses fitted weights if None)
        
        Returns:
            blended_predictions: Weighted average of predictions
        """
        if weights is None:
            weights = self.best_weights
        
        if weights is None:
            raise ValueError("Must fit ensemble or provide weights")
        
        return np.average(test_predictions, weights=weights, axis=0)


class GridSearchEnsemble:
    """Exhaustive grid search for optimal ensemble weights (small number of models)."""
    
    def __init__(
        self,
        metric: str = 'rmse',
        step: float = 0.1,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            metric: 'rmse', 'mae', 'auc', 'logloss'
            step: Grid step size (0.1 = 10 values per dimension)
            use_gpu: Whether to use GPU
            verbose: Print progress
        """
        self.metric = metric
        self.step = step
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = verbose
        
        self.best_weights = None
        self.best_score = None
        self.xp = cp if self.use_gpu else np
    
    def fit(
        self,
        predictions: List[np.ndarray],
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Find optimal weights via grid search.
        
        Args:
            predictions: List of OOF predictions
            y_true: True target values
        
        Returns:
            best_weights: Optimal weights
            best_score: Best score
        """
        from itertools import product
        
        n_models = len(predictions)
        
        if n_models > 4:
            print(f"Warning: Grid search with {n_models} models may be slow.")
            print("Consider using HillClimbingEnsemble instead.")
        
        # Move to GPU
        preds = self.xp.array([self.xp.array(p) for p in predictions])
        y = self.xp.array(y_true)
        
        # Create score function
        hc = HillClimbingEnsemble(metric=self.metric, use_gpu=self.use_gpu)
        score_func = hc._get_score_func()
        
        # Generate all weight combinations
        weight_range = np.arange(0, 1 + self.step, self.step)
        
        best_score = -np.inf
        best_weights = None
        n_combinations = len(weight_range) ** n_models
        
        if self.verbose:
            print(f"Searching {n_combinations} combinations...")
        
        for weights in tqdm(product(weight_range, repeat=n_models), 
                           total=n_combinations, disable=not self.verbose):
            weights = np.array(weights)
            
            if weights.sum() == 0:
                continue
            
            # Normalize
            weights = weights / weights.sum()
            
            # Blend and score
            weights_gpu = self.xp.array(weights)
            blended = self.xp.sum(preds * weights_gpu[:, None], axis=0)
            
            score = score_func(y, blended)
            if isinstance(score, cp.ndarray):
                score = float(score.get())
            
            if score > best_score:
                best_score = score
                best_weights = weights
        
        self.best_weights = best_weights
        self.best_score = best_score
        
        final_score = -best_score if self.metric in ['rmse', 'mae', 'logloss'] else best_score
        
        if self.verbose:
            print(f"\nGrid search best score: {final_score:.6f}")
            print(f"Best weights: {dict(enumerate(best_weights.round(4)))}")
        
        return best_weights, final_score
    
    def predict(
        self,
        test_predictions: List[np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply ensemble weights to test predictions."""
        if weights is None:
            weights = self.best_weights
        return np.average(test_predictions, weights=weights, axis=0)


def simple_average(predictions: List[np.ndarray]) -> np.ndarray:
    """Simple average ensemble (baseline)."""
    return np.mean(predictions, axis=0)


def rank_average(predictions: List[np.ndarray]) -> np.ndarray:
    """Rank-based averaging (robust to different prediction scales)."""
    from scipy.stats import rankdata
    
    ranks = [rankdata(p) / len(p) for p in predictions]
    return np.mean(ranks, axis=0)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate OOF predictions from 5 models
    n_samples = 10000
    y_true = np.random.randn(n_samples)
    
    # Create correlated predictions (simulating different models)
    oof_preds = [
        y_true + np.random.randn(n_samples) * 0.3,  # Good model
        y_true + np.random.randn(n_samples) * 0.5,  # Medium model
        y_true + np.random.randn(n_samples) * 0.7,  # Weaker model
        y_true + np.random.randn(n_samples) * 0.4,  # Good model
        y_true + np.random.randn(n_samples) * 0.6,  # Medium model
    ]
    
    print("="*50)
    print("HILL CLIMBING ENSEMBLE")
    print("="*50)
    
    # Hill climbing
    hc = HillClimbingEnsemble(
        metric='rmse',
        n_iterations=2000,
        use_gpu=True,
        verbose=True
    )
    
    weights, score = hc.fit(oof_preds, y_true)
    
    # Compare to simple average
    simple_avg = simple_average(oof_preds)
    simple_rmse = np.sqrt(np.mean((y_true - simple_avg) ** 2))
    print(f"\nSimple average RMSE: {simple_rmse:.6f}")
    print(f"Hill climbing RMSE: {score:.6f}")
    print(f"Improvement: {(simple_rmse - score) / simple_rmse * 100:.2f}%")
    
    # Generate test predictions
    test_preds = [
        np.random.randn(1000) * 0.3,
        np.random.randn(1000) * 0.5,
        np.random.randn(1000) * 0.7,
        np.random.randn(1000) * 0.4,
        np.random.randn(1000) * 0.6,
    ]
    
    final_predictions = hc.predict(test_preds)
    print(f"\nFinal predictions shape: {final_predictions.shape}")
