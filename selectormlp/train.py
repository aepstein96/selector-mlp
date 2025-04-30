import pytorch_lightning as pl
from torch.utils.data import DataLoader
from anndata import read_h5ad
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from .data import createDataset, createSVMDataset
from .models import MultiClassifier, SelectorMLP
import pickle
import json
import argparse
import os
from datetime import datetime
from .visualization import plotTrainingResults
import pandas as pd
import shutil
import torch

# Train a LinearSVC model on AnnData with logging and model saving
# Used as a baseline for comparison with SelectorMLP
def trainSVM(adata_train, adata_val, y_col, penalty="l1", loss="hinge_squared", dual=False, C=1.0):    
    # Convert AnnData to arrays
    X_train, y_train = createSVMDataset(adata_train, y_col)
    X_val, y_val = createSVMDataset(adata_val, y_col)
    
    # Train model
    model = LinearSVC(penalty=penalty.lower(), loss=loss.lower(), dual=dual, C=C)
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    train_balanced_accuracy = balanced_accuracy_score(y_train, model.predict(X_train))
    val_balanced_accuracy = balanced_accuracy_score(y_val, model.predict(X_val))
    
    # Create metrics dictionary
    metrics = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'train_balanced_accuracy': train_balanced_accuracy,
        'val_balanced_accuracy': val_balanced_accuracy,
        'penalty': penalty,
        'loss': loss,
        'dual': dual,
        'C': C
    }
    return model, metrics


# Train a SelectorMLP model using PyTorch Lightning with checkpointing and visualization
def trainSelectorMLP(adata_train, adata_val, y_col, model_config, training_config, dataloader_config, checkpoint_config, log_dir, restart=False):
    train_set = createDataset(adata_train, y_col)
    val_set = createDataset(adata_val, y_col)
        
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = dataloader_config["batch_size"]
    num_workers = dataloader_config["num_workers"]
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=dataloader_config["shuffle_train"]
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
        shuffle=dataloader_config["shuffle_val"]
    )
    
    # Model parameters
    num_features = train_set.tensors[0].shape[1]
    num_classes = len(train_set.tensors[1].unique())
    
    # Set up loggers
    base_dir = os.path.join(log_dir, f"{model_config['name']}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Create loggers
    logger1 = pl.loggers.TensorBoardLogger(save_dir=base_dir, name="")
    version = logger1.version
    logger2 = pl.loggers.CSVLogger(save_dir=base_dir, name="", version=version)
    
    # Create checkpoint directory with same version as logger
    checkpoint_dir = os.path.join(checkpoint_config['dir'], f"version_{version}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = None
    if restart:
        print("Restart enabled, looking for last checkpoint...")
        # First check if last.ckpt exists
        last_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_checkpoint_path):
            checkpoint_path = last_checkpoint_path
            print(f"Found last checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found, starting from scratch")
    
    # Initialize model from checkpoint or create new one
    if checkpoint_path:
        model = MultiClassifier.load_from_checkpoint(checkpoint_path)
    else:
        print("Training from scratch...")
        model = MultiClassifier(
                SelectorMLP, 
                num_features, 
                num_classes, 
                lr=training_config["lr"], 
                batch_norm=training_config["batch_norm"], 
                noise_std=training_config["noise_std"], 
                selector_type=model_config["selector_type"],
                weight_decay=training_config["weight_decay"], 
                dropout=training_config["dropout"],
                penalty=model_config["penalty"]
            )
    
    # Set up checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_multiclass_accuracy:.4f}",
        monitor='val_multiclass_accuracy',
        mode='max',
        save_top_k=checkpoint_config["save_top_k"],
        save_last=checkpoint_config["save_last"],
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"], 
        accelerator='gpu', 
        logger=[logger1, logger2], 
        callbacks=[checkpoint_callback],
        default_root_dir=base_dir,
        enable_progress_bar=False
    )
    
    trainer.fit(model, train_loader, [val_loader, train_eval_loader])
    
    # Get best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Best model: {best_model_path}")
        best_model = MultiClassifier.load_from_checkpoint(best_model_path)
    else:
        print("No best model found, using final model")
        best_model = model
        
    # Generate and save visualizations
    csv_logger = trainer.loggers[1]  # The CSVLogger is the second logger
    metrics_df = pd.read_csv(os.path.join(csv_logger.log_dir, "metrics.csv"))
    
    # Create and save training plots
    accuracy_plot, loss_plot = plotTrainingResults(metrics_df=metrics_df)
    out_dir = os.path.join(base_dir, f"version_{version}")
    os.makedirs(out_dir, exist_ok=True)
    accuracy_plot.figure.savefig(os.path.join(out_dir, "accuracy_plot.png"))
    loss_plot.figure.savefig(os.path.join(out_dir, "loss_plot.png"))
    
    # Convert metrics to Python values
    metrics = trainer.callback_metrics
    metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    metrics['best_model_path'] = best_model_path
    
    return best_model, metrics, out_dir

def train(config, model_type, restart=False):
    # Load data
    adata_train = read_h5ad(os.path.join(config["data"]["input_dir"], "adata_train.h5ad"))
    adata_val = read_h5ad(os.path.join(config["data"]["input_dir"], "adata_val.h5ad"))
    
    start_time = datetime.now()
    
    # Make directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config['trained_models_dir'], exist_ok=True)
    
    if model_type == "svm":
        model, metrics = trainSVM(
            adata_train=adata_train,
            adata_val=adata_val,
            y_col=config["data"]["y_column"],
            **config["svm"]
        )
        out_dir = os.path.join(config["log_dir"], "SVM")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save model
        save_path_local = os.path.join(out_dir, "best_model.pkl")
        with open(save_path_local, 'wb') as f:
            pickle.dump(model, f)
        save_path_global = os.path.join(config['trained_models_dir'], "SVM_best_model.pkl")
        with open(save_path_global, 'wb') as f:
            pickle.dump(model, f)
    elif model_type == "selector_mlp":
        model, metrics, out_dir = trainSelectorMLP(
            adata_train=adata_train,
            adata_val=adata_val,
            y_col=config["data"]["y_column"],
            model_config=config["model"],
            training_config=config["training"],
            dataloader_config=config["dataloader"],
            checkpoint_config=config["checkpoint"],
            log_dir=config["log_dir"],
            restart=restart
        )    
        # Save model
        save_path_local = os.path.join(out_dir, f"best_model.pkl")
        # Move model to CPU before saving
        model = model.cpu()
        model.save(save_path_local)
        save_path_global = os.path.join(config['trained_models_dir'], "SelectorMLP_best_model.pkl")
        model.save(save_path_global)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
        
    print("Training complete!")
    end_time = datetime.now()
    
    # Save training log with essential details
    logfile = os.path.join(out_dir, f"training_log.txt")
    with open(logfile, 'w') as f:
        f.write(f"Training start time: {start_time}\n")
        f.write(f"Training end time: {end_time}\n")
        f.write(f"Training duration: {end_time - start_time}\n")
        f.write(f"Training data: {os.path.join(config['data']['input_dir'], 'adata_train.h5ad')}\n")
        f.write(f"Validation data: {os.path.join(config['data']['input_dir'], 'adata_val.h5ad')}\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Model:\n{model}\n")
    
    print(f"Training complete! Total training duration: {end_time - start_time}")

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
    
    train(config, model_type=args.model_type, restart=args.restart) 