#!/bin/bash

# Exit if any problems occur
set -e
set -o pipefail
set -u

# Data splitting
echo "Step 1: Splitting the data..."
python -m selectormlp split_data --config configs/split_data.json
echo "Data splitting complete!"

# Training models
echo "Step 2: Training SelectorMLP model..."
python -m selectormlp train --config configs/train.json --model_type selector_mlp

echo "Step 3: Training SVM model..."
python -m selectormlp train --config configs/train.json --model_type svm

# Fixed paths for models
selector_path="trained_models/SelectorMLP_best_model.pkl"
svm_model="trained_models/SVM_best_model.pkl"

# Perform feature selection 
echo "Step 4: Performing feature selection with SelectorMLP..."
python -m selectormlp feature_selection --config configs/feature_selection.json --model_path ${selector_path} --model_type selector_mlp

echo "Step 5: Performing feature selection with SVM..."
python -m selectormlp feature_selection --config configs/feature_selection.json --model_path ${svm_model} --model_type svm

echo "Workflow completed successfully!"