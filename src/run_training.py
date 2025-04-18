#!/usr/bin/env python3
"""
Example script demonstrating how to use the simple config loader
for running a model training job.
"""

import os
import fire
from simple_config_loader import load_config


def train_model(**kwargs):
    """
    Train a model using configuration with command-line overrides
    
    Args:
        **kwargs: Command-line arguments to override config values
    """
    # Load configuration with any overrides
    config = load_config(**kwargs)
    
    # Print configuration that will be used
    print("\n=== Training Configuration ===")
    print(f"Restart: {config['restart']}")
    print(f"Use split: {config['use_split']}")
    print(f"Training epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['dataloader']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print()
    
    # Here you would typically:
    # 1. Load data based on config
    # 2. Create model based on config
    # 3. Set up training based on config
    # 4. Run training
    
    print("Starting training...")
    print(f"Would train for {config['training']['max_epochs']} epochs")
    print(f"Using batch size of {config['dataloader']['batch_size']}")
    print("Training complete!")
    
    return config


if __name__ == "__main__":
    fire.Fire(train_model) 