import json
import os
import torch
import sklearn
from sklearn.metrics import confusion_matrix

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


def getBestCheckpoint(folder):
    if not os.path.exists(folder):
        print("Checkpoint directory %s does not exist" % folder)
        return None
    
    best_checkpoints = []
    for file in os.listdir(folder):
        if file.endswith('.ckpt') and "last" not in file:
            checkpoint_file = os.path.join(folder, file)
            accuracy = float(file.split('accuracy=')[-1].split('.ckpt')[0])
            best_checkpoints.append((accuracy, checkpoint_file))
    
    if best_checkpoints:
        best_checkpoints.sort(reverse=True)
        checkpoint_path = best_checkpoints[0][1]
        print(f"Found checkpoint with accuracy {best_checkpoints[0][0]}: {checkpoint_path}")
        return checkpoint_path
    else:
        print("No checkpoints in checkpoint directory %s" % folder)
        return None


def load_config(config_path='src/config.json'):
    """
    Load configuration from JSON file
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def getLogs(folder, find_version=True):
    """
    Read logs from the specified folder
    
    Parameters:
    -----------
    folder : str
        Path to the log folder
    find_version : bool
        Whether to find the latest version folder
        
    Returns:
    --------
    tuple
        (logs_step, logs_epoch) DataFrames
    """
    import pandas as pd
    import os
    
    if find_version:
        versions = [f for f in os.listdir(folder) if f.startswith('version')]
        versions.sort()
        folder = os.path.join(folder, versions[-1])

    print("Reading logs from %s..." % folder)

    logs = pd.read_csv(os.path.join(folder, "metrics.csv"))
    logs.columns = [col.split('/')[0] for col in logs.columns]

    logs_step = logs[logs['train_batch_loss'].notnull()].dropna(axis=1)
    logs_step.set_index('step', inplace=True)
    logs_epoch = logs[logs['val_eval_loss'].notnull()].dropna(axis=1)
    logs_epoch.set_index('epoch', inplace=True)

    return logs_step, logs_epoch