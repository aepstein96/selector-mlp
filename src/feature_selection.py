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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchmetrics import Accuracy
from data import createDataset
import matplotlib.pyplot as plt
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
        for idx in chosen_features:
            mask[:, idx] = 1.0
        
        # Zero out unselected weights by multiplying with mask
        net.selector.weight.data = net.selector.weight.data * mask
        
    return chosen_features, net

# Filter features using a SVM
# Returns the accuracy of the model on the filtered data
def filterFeaturesSVM(feat_nums, indices, model, X, y):
    accs = []
    per_class_accs = []
    filter = np.zeros(indices.shape[0])
    
    # Filter features and calculate accuracy
    for num in tqdm(feat_nums):
        filter[indices[:num]] = 1
        X_filtered = np.multiply(X, filter)
        y_pred = model.predict(X_filtered)
        acc = accuracy_score(y, y_pred)
        per_class_acc = balanced_accuracy_score(y, y_pred, average=None)
        accs.append(acc)
        per_class_accs.append(per_class_acc)

    accs = np.array(accs, dtype=float)
    per_class_accs = np.array(per_class_accs, dtype=float)
    return accs, per_class_accs

# Filter features using an MLP
# Returns the accuracy of the model on the filtered data
def filterFeaturesMLP(feat_nums, indices, model, X, y):
    with torch.no_grad():
        accs = []
        per_class_accs = []
        filter = np.zeros(indices.shape[0])
        num_classes = len(torch.unique(y))
        
        # Get device of the model
        device = next(model.parameters()).device
        
        # Move metrics to the same device as the model
        accFunc = Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device)
        perClassAccFunc = Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
        
        # Filter features and calculate accuracy
        for num in tqdm(feat_nums):
            filter[indices[:num]] = 1
            # Ensure filter tensor is on the same device as X and model
            filter_tensor = torch.from_numpy(filter).bool().to(device)
            X_filtered = torch.mul(X.to(device), filter_tensor)
            y_pred = model(X_filtered)
            accs.append(accFunc(y_pred, y.to(device)))
            per_class_accs.append(perClassAccFunc(y_pred, y.to(device)))

        accs = torch.stack(accs).cpu().detach().numpy()
        per_class_accs = torch.stack(per_class_accs).cpu().detach().numpy()
        return accs, per_class_accs


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
    dataset = createDataset(adata, config['y_column'])
    
    X, y = dataset.tensors
    num_classes = len(torch.unique(y))
    
    # Perform feature selection
    if isinstance(model, SVC):
        model_type = "svm"
        weight_abs = np.abs(model.coef_).sum(axis=0)
        featureSelectionFunction = filterFeaturesSVM
        
    else:
        model_type = "SelectorMLP"
        model.eval()
        with torch.no_grad():
            if model.selector is None: # For an MLP, sum up all of the weights corresponding to each feature
                weight_abs = np.sum(np.abs(model.h1.weight.cpu().detach().numpy()), axis=0)
            else:
                weight_abs = np.abs(model.selector.weight.cpu().detach().numpy())
        
        featureSelectionFunction = filterFeaturesMLP
    
    # Perform proper feature selection and measure accuracy
    feat_nums = np.arange(0, weight_abs.shape[0], config['group_size'])
    idx_sorted = np.argsort(weight_abs)[::-1]
    accs_sorted, per_class_accs_sorted = featureSelectionFunction(feat_nums, idx_sorted, model, X, y)
    
    # Perform random feature selection and measure accuracy
    accs_random = np.zeros(feat_nums.shape[0])
    per_class_accs_random = np.zeros(feat_nums.shape[0])
    
    for i in range(config['random_reps']):
        idx_random = np.random.permutation(weight_abs.shape[0])
        accs_tmp, per_class_accs_tmp = featureSelectionFunction(feat_nums, idx_random, model, X, y)
        accs_random += accs_tmp
        per_class_accs_random += per_class_accs_tmp
        
    accs_random /= config['random_reps']
    per_class_accs_random /= config['random_reps']
    
    # Plot and save results
    os.makedirs(config['results_dir'], exist_ok=True)
    print(f"Saving results to {config['results_dir']}")
    results = pd.DataFrame({'Accuracy (selected features)': accs_sorted, 'Accuracy (random features)': accs_random, 'Mean per-class acc. (selected)': per_class_accs_sorted, 'Mean per-class acc. (random)': per_class_accs_random}, index=feat_nums)
    plt = results.plot()
    plt.set_xlabel('Number of features selected')
    plt.set_ylabel('Accuracy') 
    plt.set_ylim(0, 1)
    plt.figure.savefig(os.path.join(config['results_dir'], f"{model_type}_feature_selection_results.png"))
    results.to_csv(os.path.join(config['results_dir'], f"{model_type}_feature_selection_results.csv"), index=True)
    print("Results saved")

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