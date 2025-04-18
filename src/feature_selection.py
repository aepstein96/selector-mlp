from sklearn.metrics import log_loss
import torch
import numpy as np
from tqdm import tqdm
import torchmetrics
import torch.nn as nn

def selectFeaturesSVM(feat_nums, indices, model, X, y):
    accs = []
    filter = np.zeros(indices.shape[0])
    for num in tqdm(feat_nums):
      filter[indices[:num]] = 1
      X_filtered = np.multiply(X, filter)
      y_pred = model.predict(X_filtered)
      acc = (y_pred==y).sum() / len(y)
      accs.append(acc)

    accs = np.array(accs, dtype=float)
    return accs

def testFeatureSelectionSVM(model, X, y, group_size=50, random_reps=5, **kwargs):

    if type(X) == torch.Tensor:
      X = X.cpu().detach().numpy()
    if type(y) == torch.Tensor:
      y = y.cpu().detach().numpy()

    weight_abs = np.abs(model.coef_).sum(axis=0) # hope the sum makes sense
    feat_nums = np.arange(0, weight_abs.shape[0], group_size)

    # Get sorted weights
    idx_sorted = np.argsort(weight_abs)[::-1]
    accs_sorted = selectFeaturesSVM(feat_nums, idx_sorted, model, X, y, **kwargs)

    # Get random weight results
    accs_list = []
    accs_random = np.zeros(feat_nums.shape[0])
    for i in range(random_reps):
      idx_random = np.random.permutation(weight_abs.shape[0])
      accs_tmp = selectFeaturesSVM(feat_nums, idx_random, model, X, y, **kwargs)
      accs_random += accs_tmp

    accs_random /= random_reps
    return accs_sorted, accs_random, feat_nums

def selectFeatures(feat_nums, indices, net, X, y, lossFunc=nn.CrossEntropyLoss(), accFunc=torchmetrics.Accuracy()):
  with torch.no_grad():
    losses = []
    accs = []
    filter = np.zeros(indices.shape[0])
    for num in tqdm(feat_nums):
      filter[indices[:num]] = 1
      X_filtered = torch.mul(X, torch.from_numpy(filter).float())
      y_pred = net(X_filtered)
      losses.append(lossFunc(y_pred, y))
      accs.append(accFunc(y_pred, y))

    losses = torch.stack(losses).cpu().detach().numpy()
    accs = torch.stack(accs).cpu().detach().numpy()
    return losses, accs

def testFeatureSelection(net, X, y, group_size=50, random_reps=5, **kwargs):
  with torch.no_grad():
    if net.selector is None: # For an MLP, sum up all of the weights corresponding to each feature
      weight_abs = np.sum(np.abs(net.h1.weight.cpu().detach().numpy()), axis=0)
    else:
      weight_abs = np.abs(net.selector.weight.cpu().detach().numpy())

    feat_nums = np.arange(0, weight_abs.shape[0], group_size)

    # Get sorted weights
    idx_sorted = np.argsort(weight_abs)[::-1]
    losses_sorted, accs_sorted = selectFeatures(feat_nums, idx_sorted, net, X, y, **kwargs)

    # Get random weight results
    losses_random = np.zeros(feat_nums.shape[0])
    accs_random = np.zeros(feat_nums.shape[0])

    for i in range(random_reps):
      idx_random = np.random.permutation(weight_abs.shape[0])
      losses_tmp, accs_tmp = selectFeatures(feat_nums, idx_random, net, X, y, **kwargs)
      losses_random += losses_tmp
      accs_random += accs_tmp

    losses_random /= random_reps
    accs_random /= random_reps
    return losses_sorted, accs_sorted, losses_random, accs_random, feat_nums