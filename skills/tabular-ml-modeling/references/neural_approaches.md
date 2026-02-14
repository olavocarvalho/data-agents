# Neural Network Approaches for Tabular Data

## When to Use Neural Networks

Neural networks are useful for tabular data when:
- Dataset is very large (>100K rows)
- Need differentiable predictions (for end-to-end learning)
- Combining with other modalities (images, text)
- Want representation learning / embeddings

For most tabular tasks, gradient boosting outperforms neural networks. Use NNs as part of an ensemble rather than sole approach.

## TabPFN: Foundation Model for Small Data

TabPFN is a transformer pre-trained on synthetic tabular data. Works without training on your data.

### Best For
- Small datasets (<10,000 samples)
- Classification and regression
- Fast inference (single forward pass)
- When you can't afford hyperparameter tuning

### Limitations
- Memory-intensive for large datasets
- Limited to ~10K samples, ~500 features
- Requires GPU with 8-16GB VRAM

### Usage

```python
from tabpfn import TabPFNClassifier, TabPFNRegressor

# Classification
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Regression
reg = TabPFNRegressor()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# With custom settings
clf = TabPFNClassifier(
    N_ensemble_configurations=32,  # More = better but slower
    device='cuda'
)
```

### For Large Datasets

TabPFN doesn't scale to 7M rows, but can be used:
- As ensemble member on sampled data
- For generating meta-features

```python
def tabpfn_meta_features(X_train, y_train, X_test, n_samples=5000):
    """Use TabPFN on sample for meta-features."""
    from sklearn.model_selection import train_test_split
    
    # Sample for TabPFN
    if len(X_train) > n_samples:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=n_samples, stratify=y_train
        )
    else:
        X_sample, y_sample = X_train, y_train
    
    clf = TabPFNClassifier()
    clf.fit(X_sample, y_sample)
    
    return clf.predict_proba(X_test)
```

## TabNet: Attention-Based Deep Learning

TabNet uses attention mechanism for interpretable feature selection.

### Key Features
- Sparse feature selection (like trees)
- Self-supervised pretraining option
- Interpretable feature importance
- Works on medium-large datasets

### Installation

```bash
pip install pytorch-tabnet
```

### Usage

```python
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
import torch

# Define model
model = TabNetRegressor(
    n_d=64,           # Width of decision prediction layer
    n_a=64,           # Width of attention embedding
    n_steps=5,        # Number of sequential attention steps
    gamma=1.5,        # Coefficient for feature reusage
    n_independent=2,  # Number of independent GLU layers
    n_shared=2,       # Number of shared GLU layers
    lambda_sparse=1e-4,  # Sparsity regularization
    momentum=0.3,
    clip_value=2.0,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    device_name='cuda',
    verbose=1
)

# Training
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['rmse'],
    max_epochs=200,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Prediction
predictions = model.predict(X_test)

# Feature importance (interpretability)
importance = model.feature_importances_
```

### Self-Supervised Pretraining

```python
from pytorch_tabnet.pretraining import TabNetPretrainer

# Pretrain on unlabeled data
pretrain_model = TabNetPretrainer(
    n_d=64, n_a=64, n_steps=5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    device_name='cuda'
)

# Combine train and test for pretraining
X_combined = np.vstack([X_train, X_test])

pretrain_model.fit(
    X_train=X_combined,
    max_epochs=100,
    patience=10,
    batch_size=2048,
    virtual_batch_size=128,
    pretraining_ratio=0.8  # Mask 80% of features
)

# Save pretrained model
pretrain_model.save_model('tabnet_pretrained')

# Load for supervised fine-tuning
model = TabNetRegressor(device_name='cuda')
model.load_weights_from_pretrained(
    pretrain_model, 
    n_d=64, n_a=64, n_steps=5
)

model.fit(X_train, y_train, ...)
```

## Simple MLP Baseline

Sometimes a well-tuned MLP works surprisingly well.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

# Training function
def train_mlp(X_train, y_train, X_val, y_val, epochs=100, batch_size=1024):
    device = torch.device('cuda')
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Model
    model = TabularMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

## Entity Embeddings for Categorical Features

Learn embeddings for categorical variables, similar to word embeddings.

```python
class TabularEmbeddingModel(nn.Module):
    def __init__(self, cat_dims, num_features, embedding_dim=8, hidden_dim=128):
        """
        Args:
            cat_dims: List of (n_categories, embedding_dim) for each categorical feature
            num_features: Number of numerical features
            embedding_dim: Default embedding dimension
        """
        super().__init__()
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, min(embedding_dim, n_cats // 2 + 1))
            for n_cats in cat_dims
        ])
        
        # Total embedding dimension
        total_emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        
        # MLP layers
        self.fc1 = nn.Linear(total_emb_dim + num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x_cat, x_num):
        # Embed categorical features
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embeddings, dim=1)
        
        # Concatenate with numerical features
        x = torch.cat([x_emb, x_num], dim=1)
        
        # MLP
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        return self.fc3(x).squeeze()

# Usage
cat_dims = [10, 50, 100, 200]  # Number of categories for each cat feature
model = TabularEmbeddingModel(cat_dims, num_features=50)

# Can extract learned embeddings for use with other models
embeddings = model.embeddings[0].weight.detach().cpu().numpy()
```

## Neural Network as Ensemble Member

Best strategy: use NN as one of several diverse models.

```python
def train_diverse_nn_ensemble(X_train, y_train, X_val, y_val, n_models=3):
    """Train diverse NNs for ensembling."""
    
    predictions = []
    
    # MLP with different architectures
    architectures = [
        [256, 128, 64],
        [512, 256, 128],
        [128, 64, 32]
    ]
    
    for arch in architectures:
        model = train_mlp_with_arch(X_train, y_train, X_val, y_val, arch)
        predictions.append(model.predict(X_val))
    
    # Average predictions
    return np.mean(predictions, axis=0)
```

## Hyperparameter Recommendations

### TabNet
```python
# For large datasets (>100K rows)
params = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 5,
    'gamma': 1.5,
    'lambda_sparse': 1e-4,
    'batch_size': 4096,
    'virtual_batch_size': 256,
    'lr': 0.02
}

# For medium datasets (10K-100K rows)
params = {
    'n_d': 32,
    'n_a': 32,
    'n_steps': 3,
    'gamma': 1.3,
    'lambda_sparse': 1e-3,
    'batch_size': 1024,
    'virtual_batch_size': 128,
    'lr': 0.01
}
```

### MLP
```python
# For 2750 features
params = {
    'hidden_dims': [1024, 512, 256],  # Progressive reduction
    'dropout': 0.3,
    'batch_size': 2048,
    'lr': 1e-3,
    'weight_decay': 1e-5
}
```

## When NOT to Use Neural Networks

- Small datasets (<10K rows) - use TabPFN or gradient boosting
- When interpretability is critical - use gradient boosting
- When training time is limited - gradient boosting trains faster
- When you need reproducibility - NNs have more randomness

## Recommended Ensemble Strategy

```python
# Combine NNs with gradient boosting
predictions = []

# Gradient boosting (60% weight typically)
predictions.append(xgb_pred * 0.20)
predictions.append(lgb_pred * 0.20)
predictions.append(cat_pred * 0.20)

# Neural networks (30% weight typically)
predictions.append(tabnet_pred * 0.15)
predictions.append(mlp_pred * 0.15)

# Foundation model (10% weight)
predictions.append(tabpfn_pred * 0.10)  # On sampled data

final = sum(predictions)
```
