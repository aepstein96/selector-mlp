import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from anndata import read_h5ad
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from data import createDataset, createSVMDataset
from models import MultiClassifier, SelectorMLP
import pickle
import scanpy as sc
import json
import argparse
import os
from datetime import datetime
from utils import getBestCheckpoint

def trainSVM(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()

    # Load data
    adata_train = read_h5ad(os.path.join(config["data"]["input_dir"], "adata_train.h5ad"))
    adata_val = read_h5ad(os.path.join(config["data"]["input_dir"], "adata_val.h5ad"))
    
    X_train, y_train = createSVMDataset(adata_train, config["data"]["y_column"])
    X_val, y_val = createSVMDataset(adata_val, config["data"]["y_column"])
    
    # Train SVM with SVM-specific parameters if available
    penalty = config["svm"].get("penalty", config["model"]["reg_type"].lower())
    loss = config["svm"].get("loss", 'squared_hinge')
    dual = config["svm"].get("dual", False)
    C = config["svm"].get("C", 1.0)
    
    model = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=C)
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    train_balanced_accuracy = balanced_accuracy_score(y_train, model.predict(X_train))
    val_balanced_accuracy = balanced_accuracy_score(y_val, model.predict(X_val))
    
    # Set up loggers - use consistent base directory for all runs with the same name
    run_dir = os.path.join(config['log_dir'], f"{config['svm']['name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save logs and model with timestamp
    logfile = os.path.join(run_dir, f"training_log.txt")
    end_time = datetime.now()
    with open(logfile, 'w') as f:
        f.write(f"Training start time: {timestamp}\n")
        f.write(f"Training end time: {end_time.strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Training duration: {end_time - start_time}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Validation accuracy: {val_accuracy:.4f}\n")
        f.write(f"Train balanced accuracy: {train_balanced_accuracy:.4f}\n")
        f.write(f"Validation balanced accuracy: {val_balanced_accuracy:.4f}\n\n")
        f.write(f"SVM parameters:\n")
        f.write(f"  Penalty: {penalty}\n")
        f.write(f"  Loss: {loss}\n")
        f.write(f"  Dual: {dual}\n")
        f.write(f"  C: {C}\n\n")
        f.write(f"Config: {json.dumps(config, indent=2)}\n\n")
        f.write(f"Model: {model}\n\n")
    
    # Save model in the run directory
    save_path = os.path.join(run_dir, f"model.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"SVM training complete! Model saved at: {save_path}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    return model, val_accuracy


def trainSelectorMLP(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()

    # Load data
    print("Loading train and val sets...")        
    adata_train = read_h5ad(os.path.join(config["data"]["input_dir"], "adata_train.h5ad"))
    adata_val = read_h5ad(os.path.join(config["data"]["input_dir"], "adata_val.h5ad"))
    
    train_set = createDataset(adata_train, config["data"]["y_column"])
    val_set = createDataset(adata_val, config["data"]["y_column"])
        
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=config["dataloader"]["shuffle_train"]
    )
    
    train_eval_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=config["dataloader"]["shuffle_val"]
    )
    
    # Model parameters
    num_features = train_set.tensors[0].shape[1]
    num_classes = len(train_set.tensors[1].unique())
    
    # Set up loggers - use consistent base directory for all runs with the same name
    base_dir = os.path.join(config["log_dir"], f"{config['model']['name']}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Use timestamp for the version name in the logger
    logger1 = pl.loggers.TensorBoardLogger(save_dir=base_dir, version=timestamp)
    logger2 = pl.loggers.CSVLogger(save_dir=base_dir, version=timestamp)
    
    # Path for this run's outputs
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Check for checkpoint to restart from
    checkpoint_path = None
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if config["restart"]:
        print("Restart enabled, searching for best checkpoint...")
        
        # Look for checkpoints
        if config['checkpoint']['restart_run'] == 'last':
            # Find most recent timestamp directory
            timestamp_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            pre_checkpoint_dir = os.path.join(checkpoint_dir, max(timestamp_dirs)) if timestamp_dirs else ""
        else:
            pre_checkpoint_dir = os.path.join(checkpoint_dir, config['checkpoint']['restart_run'])
        
        checkpoint_path = getBestCheckpoint(pre_checkpoint_dir)
        
    if checkpoint_path:
        model = MultiClassifier.load_from_checkpoint(checkpoint_path)
    else:
        print("Training from scratch...")
        model = MultiClassifier(
                SelectorMLP, 
                num_features, 
                num_classes, 
                lr=config["training"]["lr"], 
                batch_norm=config["training"]["batch_norm"], 
                noise_std=config["training"]["noise_std"], 
                selector_type=config["model"]["selector_type"],
                weight_decay=config["training"]["weight_decay"], 
                dropout=config["training"]["dropout"],
                reg_type=config["model"]["reg_type"]
            )
    
    # Set up checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, timestamp),
        filename="{epoch:02d}-{val_accuracy:.4f}",
        monitor='val_accuracy',
        mode='max',
        save_top_k=config["checkpoint"]["save_top_k"],
        save_last=config["checkpoint"]["save_last"],
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"], 
        accelerator='gpu', 
        logger=[logger1, logger2], 
        callbacks=[checkpoint_callback],
        default_root_dir=base_dir
    )
    
    trainer.fit(model, train_loader, [val_loader, train_eval_loader])
    
    end_time = datetime.now()
    
    # Get validation metrics and best model path
    val_metrics = trainer.callback_metrics
    best_model_path = checkpoint_callback.best_model_path
    
    # Save log in the run directory with timestamp
    logfile = os.path.join(run_dir, f"training_log.txt")
    with open(logfile, 'w') as f:
        f.write(f"Training start time: {timestamp}\n")
        f.write(f"Training end time: {end_time.strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Training duration: {end_time - start_time}\n")
        f.write(f"Validation metrics: {val_metrics}\n\n")
        f.write(f"Best model path: {best_model_path}\n\n")
        f.write(f"Config: {json.dumps(config, indent=2)}\n\n")
        f.write(f"Model: {model}\n\n")
        if checkpoint_path and config["restart"]:
            f.write(f"Restarted from: {checkpoint_path}\n\n")
        
    # Save model with timestamp in the run directory
    save_path = os.path.join(run_dir, f"final_model.pkl")
    model.save(save_path)
        
    print(f"Training complete! Best model saved at: {best_model_path}")
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join("src", "config.json"))
    parser.add_argument("--restart", action="store_true", help="Restart training from best checkpoint")
    parser.add_argument("--model_type", type=str, choices=["svm", "selector_mlp"], default="selector_mlp",
                        help="Type of model to train (default: selector_mlp)")
    args = parser.parse_args()
    
    # Load configuration from file
    with open(args.config, 'r') as f:
        config = json.load(f)    
        
    # Override with command line arguments if provided
    if args.restart:
        config["restart"] = True
    
    # Create log directory if it doesn't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Train the selected model type
    if args.model_type == "svm":
        trainSVM(config)
    else:  # selector_mlp
        trainSelectorMLP(config) 