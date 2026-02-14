#!/usr/bin/env python3
"""
Baseline Trainer for Tabular ML
Trains XGBoost, LightGBM, CatBoost, and cuML Random Forest with CV.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
import gc
warnings.filterwarnings('ignore')

# Try to import cuML for GPU-accelerated RF
try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as cuRF
    CUML_AVAILABLE = False


class BaselineTrainer:
    """Train multiple baseline models with cross-validation."""
    
    def __init__(self, task='regression', n_folds=5, random_state=42, use_gpu=True):
        """
        Args:
            task: 'regression' or 'classification'
            n_folds: Number of CV folds
            random_state: Random seed
            use_gpu: Whether to use GPU acceleration
        """
        self.task = task
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        self.models = {}
        self.oof_preds = {}
        self.test_preds = {}
        self.cv_scores = {}
    
    def get_xgb_params(self):
        """XGBoost parameters."""
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
        }
        
        if self.use_gpu:
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'
        
        if self.task == 'regression':
            params['objective'] = 'reg:squarederror'
        else:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
        
        return params
    
    def get_lgb_params(self):
        """LightGBM parameters."""
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 255,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'feature_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        if self.use_gpu:
            params['device'] = 'gpu'
        
        if self.task == 'regression':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        else:
            params['objective'] = 'binary'
            params['metric'] = 'auc'
        
        return params
    
    def get_cat_params(self):
        """CatBoost parameters."""
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'random_seed': self.random_state,
            'verbose': 100,
        }
        
        if self.use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        
        if self.task == 'regression':
            params['loss_function'] = 'RMSE'
        else:
            params['loss_function'] = 'Logloss'
            params['eval_metric'] = 'AUC'
        
        return params
    
    def get_metric(self, y_true, y_pred):
        """Calculate appropriate metric."""
        if self.task == 'regression':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        else:
            return roc_auc_score(y_true, y_pred)
    
    def get_cv_splitter(self, y):
        """Get appropriate CV splitter."""
        if self.task == 'classification':
            return StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                                   random_state=self.random_state)
        return KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
    
    def train_xgboost(self, X, y, X_test=None, cat_features=None):
        """Train XGBoost with CV."""
        print("\n" + "="*50)
        print("Training XGBoost")
        print("="*50)
        
        params = self.get_xgb_params()
        cv = self.get_cv_splitter(y)
        
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        models = []
        
        if self.task == 'regression':
            ModelClass = xgb.XGBRegressor
        else:
            ModelClass = xgb.XGBClassifier
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = ModelClass(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
            
            if self.task == 'classification':
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                if X_test is not None:
                    test_preds += model.predict_proba(X_test)[:, 1] / self.n_folds
            else:
                oof[val_idx] = model.predict(X_val)
                if X_test is not None:
                    test_preds += model.predict(X_test) / self.n_folds
            
            models.append(model)
            
            fold_score = self.get_metric(y_val, oof[val_idx])
            print(f"Fold {fold + 1} Score: {fold_score:.6f}")
        
        cv_score = self.get_metric(y, oof)
        print(f"\nXGBoost CV Score: {cv_score:.6f}")
        
        self.models['xgboost'] = models
        self.oof_preds['xgboost'] = oof
        self.test_preds['xgboost'] = test_preds
        self.cv_scores['xgboost'] = cv_score
        
        return oof, test_preds, cv_score
    
    def train_lightgbm(self, X, y, X_test=None, cat_features=None):
        """Train LightGBM with CV."""
        print("\n" + "="*50)
        print("Training LightGBM")
        print("="*50)
        
        params = self.get_lgb_params()
        cv = self.get_cv_splitter(y)
        
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        models = []
        
        if self.task == 'regression':
            ModelClass = lgb.LGBMRegressor
        else:
            ModelClass = lgb.LGBMClassifier
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = ModelClass(**params)
            
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            
            if self.task == 'classification':
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                if X_test is not None:
                    test_preds += model.predict_proba(X_test)[:, 1] / self.n_folds
            else:
                oof[val_idx] = model.predict(X_val)
                if X_test is not None:
                    test_preds += model.predict(X_test) / self.n_folds
            
            models.append(model)
            
            fold_score = self.get_metric(y_val, oof[val_idx])
            print(f"Fold {fold + 1} Score: {fold_score:.6f}")
        
        cv_score = self.get_metric(y, oof)
        print(f"\nLightGBM CV Score: {cv_score:.6f}")
        
        self.models['lightgbm'] = models
        self.oof_preds['lightgbm'] = oof
        self.test_preds['lightgbm'] = test_preds
        self.cv_scores['lightgbm'] = cv_score
        
        return oof, test_preds, cv_score
    
    def train_catboost(self, X, y, X_test=None, cat_features=None):
        """Train CatBoost with CV."""
        print("\n" + "="*50)
        print("Training CatBoost")
        print("="*50)
        
        params = self.get_cat_params()
        cv = self.get_cv_splitter(y)
        
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        models = []
        
        if self.task == 'regression':
            ModelClass = CatBoostRegressor
        else:
            ModelClass = CatBoostClassifier
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = ModelClass(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                     cat_features=cat_features, early_stopping_rounds=50)
            
            if self.task == 'classification':
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                if X_test is not None:
                    test_preds += model.predict_proba(X_test)[:, 1] / self.n_folds
            else:
                oof[val_idx] = model.predict(X_val)
                if X_test is not None:
                    test_preds += model.predict(X_test) / self.n_folds
            
            models.append(model)
            
            fold_score = self.get_metric(y_val, oof[val_idx])
            print(f"Fold {fold + 1} Score: {fold_score:.6f}")
        
        cv_score = self.get_metric(y, oof)
        print(f"\nCatBoost CV Score: {cv_score:.6f}")
        
        self.models['catboost'] = models
        self.oof_preds['catboost'] = oof
        self.test_preds['catboost'] = test_preds
        self.cv_scores['catboost'] = cv_score
        
        return oof, test_preds, cv_score
    
    def train_random_forest(self, X, y, X_test=None, cat_features=None):
        """Train Random Forest with CV (cuML if available)."""
        print("\n" + "="*50)
        print(f"Training Random Forest ({'cuML GPU' if CUML_AVAILABLE else 'sklearn'})")
        print("="*50)
        
        cv = self.get_cv_splitter(y)
        
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = cuRF(n_estimators=100, max_depth=10, random_state=self.random_state)
            model.fit(X_tr, y_tr)
            
            oof[val_idx] = model.predict(X_val)
            if X_test is not None:
                test_preds += model.predict(X_test) / self.n_folds
            
            models.append(model)
            
            fold_score = self.get_metric(y_val, oof[val_idx])
            print(f"Fold {fold + 1} Score: {fold_score:.6f}")
        
        cv_score = self.get_metric(y, oof)
        print(f"\nRandom Forest CV Score: {cv_score:.6f}")
        
        self.models['random_forest'] = models
        self.oof_preds['random_forest'] = oof
        self.test_preds['random_forest'] = test_preds
        self.cv_scores['random_forest'] = cv_score
        
        return oof, test_preds, cv_score
    
    def train_all(self, X, y, X_test=None, cat_features=None):
        """Train all baseline models."""
        self.train_xgboost(X, y, X_test, cat_features)
        gc.collect()
        
        self.train_lightgbm(X, y, X_test, cat_features)
        gc.collect()
        
        self.train_catboost(X, y, X_test, cat_features)
        gc.collect()
        
        self.train_random_forest(X, y, X_test, cat_features)
        gc.collect()
        
        # Summary
        print("\n" + "="*50)
        print("BASELINE SUMMARY")
        print("="*50)
        for name, score in sorted(self.cv_scores.items(), key=lambda x: x[1]):
            print(f"{name}: {score:.6f}")
        
        return self.oof_preds, self.test_preds, self.cv_scores


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    
    print("Generating sample data...")
    X, y = make_regression(n_samples=10000, n_features=100, noise=0.1, random_state=42)
    X_test = X[:1000]  # Use subset as test for demo
    
    trainer = BaselineTrainer(task='regression', n_folds=3, use_gpu=True)
    oof_preds, test_preds, cv_scores = trainer.train_all(X, y, X_test)
    
    print("\nDone!")
