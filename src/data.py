import torch
from torch.utils.data import TensorDataset
import numpy as np
import os
import scipy.sparse

# Convert AnnData to numpy arrays for SVM processing
def createSVMDataset(adata, y_col):
    # Convert AnnData to arrays
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    y = adata.obs[y_col]
    if y.dtype.name == 'category':
        y = y.cat.codes.values
    else:
        y = y.values
    return X, y

# Convert AnnData to PyTorch TensorDataset
def createDataset(adata, y_col):
    # Convert AnnData to tensors
    X = torch.from_numpy(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X).float()
    # Handle both categorical and non-categorical y
    y = adata.obs[y_col]
    if y.dtype.name == 'category':
        y = y.cat.codes.values.copy()
    else:
        y = y.values.copy()
    y = torch.from_numpy(y).long()
    return torch.utils.data.TensorDataset(X, y)


# Split AnnData into train/val/test sets with balanced class distribution
# Classes are balanced to the extent possible by limiting the number of cells per class
def evenClusters(adata, col, max_train=5000, max_val=1000, max_test=1000, shuffle=True, shuffle_seed=42):
    train_cells = []
    val_cells = []
    test_cells = []
    max_total_cells = max_train + max_val + max_test
    for cluster, num_cells in adata.obs[col].value_counts().items():
        cluster_cells = adata.obs.index[adata.obs[col]==cluster].values
        # Shuffle cells within each cluster for better randomization
        if shuffle:
            np.random.seed(shuffle_seed)
            np.random.shuffle(cluster_cells)
            
        cell_multiplier = min(1, num_cells/max_total_cells)
        train = int(max_train*cell_multiplier)
        val = int(max_val*cell_multiplier)
        test = int(max_test*cell_multiplier)

        train_cells.append(cluster_cells[:train])
        val_cells.append(cluster_cells[train:train+val])
        test_cells.append(cluster_cells[train+val:train+val+test])

    adata_train = adata[np.concatenate(train_cells), :]
    adata_val = adata[np.concatenate(val_cells), :]
    adata_test = adata[np.concatenate(test_cells), :]

    return adata_train, adata_val, adata_test

# Select specific features from datasets and return new filtered datasets 
def chooseFeatures(features, *datasets):
    out_list = []
    for dataset in datasets:
        X, y = dataset.tensors
        out_list.append(TensorDataset(X[:, features.copy()], y))
    return out_list