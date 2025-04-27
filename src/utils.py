import json
import os
import torch
import sklearn
from sklearn.metrics import confusion_matrix

# Get accuracy (overall and mean per-class) for a model/dataset pair
def getTestAccuracy(model, dataset, features=None):
    X, y = dataset.tensors
    if features is not None:
        filter = torch.zeros(X.shape[1])
        filter[features.copy()] = 1
        X = torch.mul(X, filter)

    y = y.cpu().detach().numpy()

    if type(model) == sklearn.svm._classes.LinearSVC:
        X = X.cpu().detach().numpy()
        y_pred = model.predict(X)
    else:
        with torch.no_grad():
            model.eval()
            y_pred = torch.softmax(model(X), dim=1).cpu().detach().numpy().argmax(axis=1)
    accuracy = (y==y_pred).sum()/y.shape[0]
    per_class_accuracy = confusion_matrix(y_pred, y, normalize='true').diagonal()
    return accuracy, per_class_accuracy.mean()


# Find best checkpoint in a directory based on accuracy value in filename
def getBestCheckpoint(folder):
    if not os.path.exists(folder):
        print(f"Checkpoint directory {folder} does not exist")
        return None
    
    best_checkpoints = []
    for file in os.listdir(folder):
        if file.endswith('.ckpt'):
            checkpoint_file = os.path.join(folder, file)
            if "accuracy" in file:
                accuracy = float(file.split('accuracy=')[-1].split('.ckpt')[0])        
                best_checkpoints.append((accuracy, checkpoint_file))
    
    if best_checkpoints:
        best_checkpoints.sort(reverse=True)
        checkpoint_path = best_checkpoints[0][1]
        print(f"Found checkpoint with accuracy {best_checkpoints[0][0]}: {checkpoint_path}")
        return checkpoint_path
    
    # If no accuracy checkpoints found, try using the last checkpoint
    last_checkpoint = os.path.join(folder, "last.ckpt")
    if os.path.exists(last_checkpoint):
        print(f"Using last checkpoint: {last_checkpoint}")
        return last_checkpoint
    
    print(f"No checkpoints in checkpoint directory {folder}")
    return None


# Load configuration from JSON file
def load_config(config_path='src/config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Read and parse training logs from specified folder
# If find_version is True, automatically finds the latest version directory
def getLogs(folder, find_version=False):
    import pandas as pd
    import os
    
    if find_version:
        versions = [f for f in os.listdir(folder) if f.startswith('version')]
        versions.sort()
        if versions:
            folder = os.path.join(folder, versions[-1])
        else:
            # Look for "current" directory 
            current_dir = os.path.join(folder, "current")
            if os.path.exists(current_dir):
                folder = current_dir

    print("Reading logs from %s..." % folder)

    metrics_path = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return None, None
        
    logs = pd.read_csv(metrics_path)
    logs.columns = [col.split('/')[0] for col in logs.columns]

    # Split logs into per-step and per-epoch dataframes
    logs_step = logs[logs['train_batch_loss'].notnull()].dropna(axis=1)
    logs_step.set_index('step', inplace=True)
    logs_epoch = logs[logs['val_eval_loss'].notnull()].dropna(axis=1)
    logs_epoch.set_index('epoch', inplace=True)

    return logs_step, logs_epoch