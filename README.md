# SelectorMLP

A PyTorch-based feature selection and classification framework for high-dimensional biological data such as single-cell gene expression.

## Introduction

This is 

SelectorMLP combines the power of neural networks with feature selection capabilities to identify important features (e.g., genes) in high-dimensional datasets while performing classification tasks. The framework includes:

- A specialized MLP architecture with a feature selection layer
- Comparison with baseline SVM models
- Tools for evaluating feature importance
- Support for AnnData objects commonly used in single-cell analysis
- PyTorch Lightning integration for efficient training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gene-selector-mlp.git
cd gene-selector-mlp

# Create conda environment
conda create -n selector-mlp python=3.9
conda activate selector-mlp

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Key Components

- **SelectorMLP**: Neural network with a feature selection layer that identifies important features
- **Data Preprocessing**: Tools for splitting AnnData objects into balanced train/val/test sets
- **Feature Selection**: Methods to evaluate model performance with varying subsets of features
- **Visualization**: Functions to plot training metrics and feature selection results

## Usage

### Data Preprocessing

Split your AnnData object into train/validation/test sets:

```bash
python src/split_data.py --config configs/split_data.json
```

Example configuration (`configs/split_data.json`):
```json
{
    "adata_path": "raw_data/adata.h5ad",
    "y_column": "Main_cluster_name",
    "train_cluster_size": 5000,
    "val_cluster_size": 1000,
    "test_cluster_size": 1000,
    "save_dir": "intermediate_files/split_data",
    "shuffle": true,
    "shuffle_seed": 42
}
```

### Model Training

Train a SelectorMLP model:

```bash
python src/train.py --config configs/train.json --model_type selector_mlp
```

Train a baseline SVM model:

```bash
python src/train.py --config configs/train.json --model_type svm
```

Example configuration (`configs/train.json`):
```json
{
    "restart": false,
    "log_dir": "logs",
    "data": {
        "input_dir": "intermediate_files/split_data",
        "y_column": "Main_cluster_name"
    },
    "dataloader": {
        "batch_size": 256,
        "num_workers": 4,
        "shuffle_train": true,
        "shuffle_val": false
    },
    "model": {
        "name": "SelectorMLP",
        "selector_type": "std",
        "reg_type": "L1"
    },
    "training": {
        "lr": 0.0003,
        "weight_decay": 0.3,
        "dropout": 0.3,
        "batch_norm": false,
        "noise_std": 0.2,
        "max_epochs": 50
    }
}
```

### Feature Selection

Evaluate feature importance:

```bash
python src/feature_selection.py --config configs/feature_selection.json --model_path logs/SelectorMLP
```

Example configuration (`configs/feature_selection.json`):
```json
{
  "model_path": "logs/SelectorMLP",
  "adata_input": "intermediate_files/split_data/adata_val.h5ad",
  "y_column": "Main_cluster_name",
  "group_size": 50,
  "random_reps": 5,
  "results_dir": "feature_selection_results"
}
```

### Complete Workflow

Run the entire pipeline:

```bash
./example_workflow.sh
```

## Model Details

### SelectorMLP Architecture

The SelectorMLP model consists of:

1. **Selector Layer**: A specialized layer that applies weights to input features
2. **MLP Backbone**: Three fully-connected layers with batch normalization and dropout
3. **Regularization**: L1 or L2 regularization on selector weights to encourage sparsity

Selector types:
- `std`: Standard feature selector with unclamped weights
- `clamped`: Feature selector with weights clamped between 0 and 1
- `none`: No feature selection (standard MLP)

### Feature Selection Process

1. Train a SelectorMLP model on your data
2. Extract feature importance weights from the selector layer
3. Rank features by their absolute weight values
4. Evaluate model performance using different numbers of top-ranked features
5. Compare with randomly selected features as a baseline

## HPC Submission

For large datasets, you can submit jobs to an HPC cluster:

```bash
sbatch --partition=hpc_a10_a \
       --time=1:00:00 \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=4 \
       --mem=128G \
       --gpus=a10:1 \
       --job-name=SelectorMLP_train \
       --output=logs/SelectorMLP/train.out \
       --error=logs/SelectorMLP/train.err \
       --wrap="source ~/.bashrc && conda activate torch && python src/train.py --config configs/train.json"
```

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@software{SelectorMLP,
  author = {Your Name},
  title = {SelectorMLP: Feature Selection with Neural Networks},
  year = {2023},
  url = {https://github.com/yourusername/gene-selector-mlp}
}
```
