import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
from .models import MultiClassifier
import pickle
from anndata import read_h5ad
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchmetrics import Accuracy
from .data import createDataset
from .visualization import plotFeatureSelectionResults

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
        per_class_acc = balanced_accuracy_score(y, y_pred)
        accs.append(acc)
        per_class_accs.append(per_class_acc)

    accs = np.array(accs, dtype=float)
    per_class_accs = np.array(per_class_accs, dtype=float)
    return accs, per_class_accs

# Filter features using an MLP
# Returns the accuracy of the model on the filtered data
def filterFeaturesMLP(feat_nums, indices, model, X, y):
    model.eval()
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


# Main feature selection test function that evaluates model performance with different feature subsets
# Compares performance when using top features vs. randomly selected features
def testFeatureSelection(config, model_type, model_path):
    if model_type == 'svm':
        model = pickle.load(open(model_path, 'rb'))
        weight_abs = np.abs(model.coef_).sum(axis=0)
        featureSelectionFunction = filterFeaturesSVM
        
    elif model_type == 'selector_mlp':
        if os.path.splitext(model_path)[1] == '.ckpt':
            model = MultiClassifier.load_from_checkpoint(model_path).net
        else:
            model = torch.load(model_path)
            
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        with torch.no_grad():
            if model.selector is None: # For an MLP, sum up all of the weights corresponding to each feature
                weight_abs = np.sum(np.abs(model.h1.weight.cpu().detach().numpy()), axis=0)
            else:
                weight_abs = np.abs(model.selector.weight.cpu().detach().numpy())
        
        featureSelectionFunction = filterFeaturesMLP
    else:
        raise ValueError("Invalid model type (must be svm or selector_mlp)")
    
    # Load data
    adata = read_h5ad(config['adata_input'])
    dataset = createDataset(adata, config['y_column'])
    
    X, y = dataset.tensors
    
    # Perform proper feature selection and measure accuracy
    feat_nums = np.arange(0, weight_abs.shape[0], config['group_size'])
    idx_sorted = np.argsort(weight_abs)[::-1]
    accs_sorted, per_class_accs_sorted = featureSelectionFunction(feat_nums, idx_sorted, model, X, y)
    
    # Perform random feature selection and measure accuracy
    accs_random = np.zeros(feat_nums.shape[0])
    per_class_accs_random = np.zeros(feat_nums.shape[0])
    
    # Run multiple random trials and average results
    for i in range(config['random_reps']):
        idx_random = np.random.permutation(weight_abs.shape[0])
        accs_tmp, per_class_accs_tmp = featureSelectionFunction(feat_nums, idx_random, model, X, y)
        accs_random += accs_tmp
        per_class_accs_random += per_class_accs_tmp
    
    accs_random /= config['random_reps']
    per_class_accs_random /= config['random_reps']
    
    # Plot and save results
    print(f"Making plots and saving results to {config['results_dir']}")
    os.makedirs(config['results_dir'], exist_ok=True)
    results = pd.DataFrame({
        'Accuracy (selected features)': accs_sorted, 
        'Accuracy (random features)': accs_random, 
        'Mean per-class acc. (selected)': per_class_accs_sorted, 
        'Mean per-class acc. (random)': per_class_accs_random
    }, index=feat_nums)
    results_plot = plotFeatureSelectionResults(results)
    results_plot.figure.savefig(os.path.join(config['results_dir'], f"{model_type}_feature_selection_results.png"))
    results.to_csv(os.path.join(config['results_dir'], f"{model_type}_feature_selection_results.csv"), index=True)
    
    # Obtain, plot and save ranked gene list
    gene_weights = pd.DataFrame({
        'gene': adata.var.index[idx_sorted],
        'weight_abs': weight_abs[idx_sorted],
        'idx': idx_sorted
    })
    gene_weights.to_csv(os.path.join(config['results_dir'], f"{model_type}_ranked_genes.csv"), index=False)
    gene_weight_plot = gene_weights.plot(y='weight_abs')
    gene_weight_plot.set_xlabel('Gene index')
    gene_weight_plot.set_ylabel('Absolute weight')
    gene_weight_plot.figure.savefig(os.path.join(config['results_dir'], f"{model_type}_ranked_genes.png"))
    print("Results saved")
    
    return results, gene_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model (.ckpt or .pkl/.pickle file)")
    parser.add_argument("--model_type", type=str, default=None, help="Model type (svm or selector_mlp)")
    parser.add_argument("--config", type=str, default="configs/feature_selection.json", help="Path to the evaluation configuration file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    testFeatureSelection(config, args.model_type, args.model_path)