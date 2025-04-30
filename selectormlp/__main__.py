import argparse
import sys
import json
from .split_data import splitData
from .train import train
from .feature_selection import testFeatureSelection


def main():
    parser = argparse.ArgumentParser(
        description="GeneSelector: PyTorch-based feature selection and classification framework for biological data"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # Split data command
    split_parser = subparsers.add_parser("split_data", help="Split data into train/val/test sets")
    split_parser.add_argument("--config", type=str, default="configs/split_data.json", 
                              help="Path to data splitting configuration file")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, default="configs/train.json",
                              help="Path to training configuration file")
    train_parser.add_argument("--model_type", type=str, choices=["selector_mlp", "svm"], 
                              required=True, help="Type of model to train (selector_mlp or svm)")
    train_parser.add_argument("--restart", action="store_true", 
                              help="Restart training from last checkpoint (selector_mlp only)")
    
    # Feature selection command
    fs_parser = subparsers.add_parser("feature_selection", help="Perform feature selection")
    fs_parser.add_argument("--config", type=str, default="configs/feature_selection.json",
                          help="Path to feature selection configuration file")
    fs_parser.add_argument("--model_path", type=str, required=True,
                          help="Path to the trained model file")
    fs_parser.add_argument("--model_type", type=str, choices=["selector_mlp", "svm"],
                          required=True, help="Type of model for feature selection (selector_mlp or svm)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == "split_data":
        with open(args.config, "r") as f:
            config = json.load(f)
        print("Step 1: Splitting the data...")
        splitData(config)
        print("Data splitting complete!")
        
    elif args.command == "train":
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Training {args.model_type} model...")
        train(config, args.model_type, args.restart)
        print(f"Training {args.model_type} model complete!")
        
    elif args.command == "feature_selection":
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Performing feature selection with {args.model_type}...")
        testFeatureSelection(config, args.model_type, args.model_path)
        print(f"Feature selection with {args.model_type} complete!")


if __name__ == "__main__":
    main()
