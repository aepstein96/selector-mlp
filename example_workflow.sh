#!/bin/bash

# Data splitting
echo "Step 1: Splitting the data..."
#python src/split_data.py --config configs/split_data.json
echo "Data splitting complete!"

# Training models
echo "Step 2: Training SelectorMLP model..."
#python src/train.py --config configs/train.json --model_type selector_mlp

echo "Step 3: Training SVM model..."
#python src/train.py --config configs/train.json --model_type svm

# Fixed paths for models
selector_path="checkpoints/current.ckpt"
svm_model="logs/SVM/model.pkl"

# Perform feature selection 
echo "Step 4: Performing feature selection with SelectorMLP..."
python src/feature_selection.py --config configs/feature_selection.json --model_path ${selector_path}

echo "Step 5: Performing feature selection with SVM..."
#python src/feature_selection.py --config configs/feature_selection.json --model_path ${svm_model}

echo "Workflow completed successfully!"