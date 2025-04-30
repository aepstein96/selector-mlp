# SelectorMLP

A PyTorch-based feature selection and classification framework for high-dimensional biological data such as single-cell gene expression.

## About

SelectorMLP combines the power of neural networks with feature selection capabilities to identify important features (e.g., genes) in high-dimensional datasets while performing classification tasks. The basic question is, "in a single-cell RNA sequencing dataset, which genes are most important for separating cells by type?" To answer this question, I created a specialized MLP architectureto classify cells by type with a feature selection layer. The feature selection layer learns to assign weights to each feature during training, which can be encouraged to be sparse using L1 regularization. Then gene features with highest magnitude weights are assumed to be most important. By sorting the genes by weights, any arbitrary number of features can be selected.

![Feature Selection Layer Architecture](docs/architecture.png)

- Comparison with baseline SVM models
- Tools for evaluating feature importance
- Support for AnnData objects commonly used in single-cell analysis
- PyTorch Lightning integration for efficient training

The example data is a cleaned version of the human brain dataset from Sziraki *et al* (2023) (https://www.nature.com/articles/s41588-023-01572-y), subset to 100,000 cells and the 2,000 most highly variable features. Cell types were separated using unsupervised clustering and annotated manually. Annotations are stored in the column adata.obs['Main_cluster_name']; the model treats these as ground truth, as the goal is to discover the features most useful for decoding these annotations.

This is an updated and refined version of my 2022 final project of my Deep Learning class (CS 5787) at Cornell Tech. 

Originally it was a Colab notebook; I have refactored it for easier installation and use. The original report, which contains much of the original code, is included at docs/report.pdf. The report also includes more information on the model architecture. This version contains all necessary code to train the models shown in the report; for some figures, config files must be altered or additional visualizations produced.

After I completed my project, I encountered a similar solution by Covert *et al.* They employ a similar supervised method to mine, and also an unsupervised one based on an autoencoder-like structure. They make a few different choices including binarizing the scRNA-seq input data. I encourage you to check out their work as well!
- PERSIST publication: https://www.nature.com/articles/s41467-023-37392-1
- PERSIST GitHub: https://github.com/iancovert/persist

## Installation

First, clone the repository:
```bash
git clone https://github.com/yourusername/gene-selector-mlp.git
cd gene-selector-mlp
```

Then, install necessary packages. You can use pip with development mode:
```
pip install -e .'
```
Or install dependencies directly:
```
pip install -r requirements.txt
```

It's strongly suggested to install all packages in a virtual environment using either venv or conda.

## Usage

An example workflow using the existing dataset (raw_data/adata.h5ad.gz) is available at example_workflow.sh. The code is reproduced below.
It consists of several steps:
- Split the AnnData object into train, validation and test sets using split_data.py (config: configs/split_data.json). The splitting process balances classes by putting caps on the number of cells from each class/cluster that can be assigned to each of the train, validation, and test sets. There is also an option min_cluster_size to filter classes that have too few cells.
- Train SelectorMLP and SVM models on the train and validation sets (config: configs/train.json). Configuration options include training parameters, model parameters, logging, checkpointing, etc. For the SelectorMLP, training loss curves are saved to the corresponding log folder. Best models are also saved to trained_models_dir.
- Perform gene feature selection using the trained models (config: configs/feature_selection.json). The model tests accuracy and mean per-class accuracy on the test set (by default; the validation set can also be used) with different numbers of masked features at intervals specified by the group_size parameter. Features are masked in order of their corresponding weights (features with lowest weights are masked first). It also tests random orderings of features (averaged across several random orderings) for comparison. For both the SVM and the SelectorMLP, ordered feature selection should substantially out-perform random feature selection; the SelectorMLP also out-performs the SVM.

Example output figures identical to the ones this code should generate with the given data and configuration are present in example_figures.

## License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.
