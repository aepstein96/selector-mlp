import torch
from torch.utils.data import TensorDataset
import numpy as np
import os

# Convert AnnData to numpy arrays for SVM processing
def createSVMDataset(adata, y_column):
    X = adata.X.toarray()
    y = adata.obs[y_column]
    if y.dtype.name == 'category': # categorical data, e.g. clusters
        y = y.cat.codes.values
    else:
        y = y.values
    return X, y

# Convert AnnData to PyTorch TensorDataset
def createDataset(adata, y_column):
    X = torch.from_numpy(adata.X.toarray())
    y = adata.obs[y_column]
    if y.dtype.name == 'category': # categorical data, e.g. clusters
        y = torch.from_numpy(y.cat.codes.values).long()
    else:
        y = torch.from_numpy(y.values).long()
        
    return TensorDataset(X, y)

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

# Split data, create datasets, and optionally save to disk
def splitData(adata, y_column, sparse_type='dense', save_dir=None, **kwargs):
    adata_train, adata_val, adata_test = evenClusters(adata, y_column, **kwargs)
    train_set = createDataset(adata_train, y_column, sparse_type)
    val_set = createDataset(adata_val, y_column, sparse_type)
    test_set = createDataset(adata_test, y_column, sparse_type)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_set, os.path.join(save_dir, 'train.pt'))
        torch.save(val_set, os.path.join(save_dir, 'val.pt'))
        torch.save(test_set, os.path.join(save_dir, 'test.pt'))

    return train_set, val_set, test_set

# Select specific features from datasets and return new filtered datasets 
def chooseFeatures(features, *datasets):
    out_list = []
    for dataset in datasets:
        X, y = dataset.tensors
        out_list.append(TensorDataset(X[:, features.copy()], y))
    return out_list