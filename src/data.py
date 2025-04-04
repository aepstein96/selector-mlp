

def createDataset(adata, y_column, sparse=True, sparse_type='CSR'):
  if sparse:
    if sparse_type == 'CSR':
      X = scipy.sparse.csr_matrix(adata.X)
      X = torch.sparse_csr_tensor(X.indptr, X.indices, X.data, X.shape, requires_grad=True)
    elif sparse_type == 'COO':
      X = scipy.sparse.coo_matrix(adata.X)
      indices = np.vstack((X.row, X.col))
      X = torch.sparse_coo_tensor(indices, X.data, X.shape, requires_grad=True)
    else:
      raise ValueError("Valid sparse_type is 'CSR' or 'COO'.")
  else:
    X = torch.from_numpy(adata.X.toarray())
  y_col = adata.obs[y_column]
  if y_col.dtype.name == 'category': # categorical data, e.g. clusters
    y = torch.from_numpy(y_col.cat.codes.values).long()
  else:
    y = torch.from_numpy(y_col.values).long()
  return TensorDataset(X, y)

# Balancing classes
def evenClusters(adata, col, max_train=5000, max_val=1000, max_test=1000):
  train_cells = []
  val_cells = []
  test_cells = []
  max_total_cells = max_train + max_val + max_test
  for cluster, num in adata_full.obs[col].value_counts().iteritems():
    cluster_cells = adata.obs.index[adata.obs[col]==cluster].values
    num_cells = cluster_cells.shape[0]
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

def chooseFeatures(features, *datasets):
  out_list = []
  for dataset in datasets:
    X, y = dataset.tensors
    out_list.append(TensorDataset(X[:, features.copy()], y))
  return out_list