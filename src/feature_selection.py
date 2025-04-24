from sklearn.metrics import log_loss
import torch
import numpy as np
from tqdm import tqdm
import torchmetrics
import torch.nn as nn
import os
import argparse
import json
from utils import getBestCheckpoint
from models import MultiClassifier
import pickle
from anndata import read_h5ad
from sklearn.svm import SVC
import pandas as pd
# Select features using a selector MLP
# Must have a selector layer (will not work with regular MLP)
# In this case, feature selection can be done by setting the weights of the unselected features to zero within the model
# Returns the chosen features and the modified model
def selectFeaturesSelectorMLP(num_features, net):
  weight_abs = np.abs(net.selector.weight.cpu().detach().numpy())
  sorted_indices = np.argsort(weight_abs)[::-1]
  chosen_features = sorted_indices[:num_features]
  # Set unselected weights to zero in the selector layer
  with torch.no_grad():
    # Create mask of zeros with ones at chosen feature indices
    mask = torch.zeros_like(net.selector.weight)
    mask[chosen_features] = 1.0
    
    # Zero out unselected weights by multiplying with mask
    net.selector.weight.data = net.selector.weight.data * mask
    
  return chosen_features, net

# Filter features using a SVM
# Returns the accuracy of the model on the filtered data
def filterFeaturesSVM(feat_nums, indices, model, X, y):
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

    weight_abs = np.abs(model.coef_).sum(axis=0) # For SVM, sum up the absolute values of the coefficients
    feat_nums = np.arange(0, weight_abs.shape[0], group_size)

    # Get results from sorted weights
    idx_sorted = np.argsort(weight_abs)[::-1]
    accs_sorted = filterFeaturesSVM(feat_nums, idx_sorted, model, X, y, **kwargs)

    # Get random weight results (used as a baseline--how well does random selection perform?)
    accs_list = []
    accs_random = np.zeros(feat_nums.shape[0])
    for i in range(random_reps):
      idx_random = np.random.permutation(weight_abs.shape[0])
      accs_tmp = filterFeaturesSVM(feat_nums, idx_random, model, X, y, **kwargs)
      accs_random += accs_tmp

    accs_random /= random_reps
    return accs_sorted, accs_random, feat_nums

def filterFeatures(feat_nums, indices, net, X, y, lossFunc=nn.CrossEntropyLoss(), accFunc=torchmetrics.Accuracy()):
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
    losses_sorted, accs_sorted = filterFeatures(feat_nums, idx_sorted, net, X, y, **kwargs)

    # Get random weight results (used as a baseline--how well does random selection perform?)
    losses_random = np.zeros(feat_nums.shape[0])
    accs_random = np.zeros(feat_nums.shape[0])

    for i in range(random_reps):
      idx_random = np.random.permutation(weight_abs.shape[0])
      losses_tmp, accs_tmp = filterFeatures(feat_nums, idx_random, net, X, y, **kwargs)
      losses_random += losses_tmp
      accs_random += accs_tmp

    losses_random /= random_reps
    accs_random /= random_reps
    return losses_sorted, accs_sorted, losses_random, accs_random, feat_nums
  
  
def testFeatureSelection(config):
  # Load model    
  if os.path.isdir(config['model_path']):
      model_path = getBestCheckpoint(config['model_path'])
      if model_path is None:
          raise ValueError("No model found in checkpoint directory")
      model = MultiClassifier.load_from_checkpoint(model_path).net
      
  elif os.path.splitext(config['model_path'])[1] == '.pkl' or os.path.splitext(config['model_path'])[1] == '.pickle':
      model = pickle.load(open(config['model_path'], 'rb'))
      
  elif os.path.splitext(config['model_path'])[1] == '.ckpt':
      model = MultiClassifier.load_from_checkpoint(config['model_path']).net
      
  else:
      raise ValueError("Invalid model path")
  
  # Load data
  adata = read_h5ad(config['adata_input'])
  X = adata.X.toarray()
  y = adata.obs[config['y_column']]
  
  # Perform feature selection
  if isinstance(model, SVC):
     model_type = "svm"
     accs_sorted, accs_random, feat_nums = testFeatureSelectionSVM(model, X, y, config['group_size'], config['random_reps'])
     results = pd.DataFrame({'accs_sorted': accs_sorted, 'accs_random': accs_random}, index=feat_nums)
     
  else:
     model_type = "SelectorMLP"
     losses_sorted, accs_sorted, losses_random, accs_random, feat_nums = testFeatureSelection(model, X, y, config['group_size'], config['random_reps'])
     results = pd.DataFrame({'accs_sorted': accs_sorted, 'accs_random': accs_random, 'losses_sorted': losses_sorted, 'losses_random': losses_random}, index=feat_nums)
    
  # Save results
  plt = results.plot()  
  plt.figure.savefig(os.path.join(config['results_dir'], f"{model_type}_feature_selection_results.png"))
  results.to_csv(os.path.join(config['results_dir'], f"{model_type}_feature_selection_results.csv"), index=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, default=None, help="Path to the model (folder of .ckpt, or .ckpt or .pkl/.pickle file)")
  parser.add_argument("--config", type=str, default="configs/feature_selection.json", help="Path to the evaluation configuration file")
  args = parser.parse_args()
  
  with open(args.config, "r") as f:
      config = json.load(f)
  
  if args.model_path:
      config['model_path'] = args.model_path
  
  testFeatureSelection(config)