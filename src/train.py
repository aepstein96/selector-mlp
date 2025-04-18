import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils import load_config
from data import splitData
from models import MultiClassifier, SelectorMLP, Selector, ClampedSelector

import scanpy as sc
import json
import argparse
import os

def main(config):
    # Load data
    
    if config["use_split"]:
        print("Loading train and val sets...")        
        train_set = torch.load(os.path.join(config["data"]["intermediate_path"], "train.pt"))
        val_set = torch.load(os.path.join(config["data"]["intermediate_path"], "val.pt"))
        #test_set = torch.load(os.path.join(config["data"]["intermediate_path"], "test.pt"))
        
    else:
        print("Loading data...")
        adata = sc.read_h5ad(config["data"]["adata_path"])
        
        print("Splitting data...")
        train_set, val_set, test_set = splitData(adata, config["data"]["column"], config["data"]["torch_sparsity_type"], os.path.join(config["data"]["intermediate_path"], "split_data"))
        
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
    training_config = config["training"]
    if config["model"]["selector_type"] == "clamped":
        selector_layer = ClampedSelector
    elif config["model"]["selector_type"] == "std":
        selector_layer = Selector
    elif config["model"]["selector_type"] == "none":
        selector_layer = None
        
    model = MultiClassifier(
        SelectorMLP, 
        num_features, 
        num_classes, 
        lr=config["training"]["lr"], 
        batch_norm=config["training"]["batch_norm"], 
        noise_std=config["training"]["noise_std"], 
        SelectorLayer=selector_layer,
        weight_decay=config["training"]["weight_decay"], 
        dropout=config["training"]["dropout"]
    )
        
    # Set up loggers
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    logger1 = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=config["logging"]["name"])
    logger2 = pl.loggers.CSVLogger(save_dir=save_dir, name=config["logging"]["name"])
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"], 
        accelerator='gpu', 
        logger=[logger1, logger2], 
        default_root_dir=save_dir
    )
    
    trainer.fit(model, train_loader, [val_loader, train_eval_loader])
        
    # Save model
    torch.save(model.net, os.path.join(save_dir, f"{config['logging']['name']}.pkl"))
        
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join("src", "config.json"))
    parser.add_argument("--restart", type=bool, default=None)
    parser.add_argument("--split", type=bool, default=None)
    args = parser.parse_args()
    
    # Load configuration from file
    with open(args.config, 'r') as f:
        config = json.load(f)    
        
    # Override with command line arguments if provided
    if args.restart is not None:
        config["restart"] = args.restart
    
    if args.split is not None:
        config["use_split"] = args.split
    
    # Call main with the updated configuration
    main(config) 