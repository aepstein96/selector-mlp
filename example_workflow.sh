#!/bin/bash

# Set up error handling
set -e
set -o pipefail

# Create necessary directories
mkdir -p logs/SelectorMLP
mkdir -p logs/SVM
mkdir -p intermediate_files/split_data
mkdir -p results/feature_selection
mkdir -p checkpoints

# Step 1: Split the data
echo "Step 1: Splitting the data..."
python src/split_data.py --config configs/split_data.json
echo "Data splitting complete!"

# Step 2: Train SelectorMLP 
echo "Step 2: Training SelectorMLP model..."
python src/train.py --config configs/train.json --model_type selector_mlp
echo "SelectorMLP training complete!"

# Step 3: Train SVM
echo "Step 3: Training SVM model..."
python src/train.py --config configs/train.json --model_type svm
echo "SVM training complete!"

# Find the most recent SelectorMLP and SVM model directories
SELECTOR_DIR=$(find logs/SelectorMLP -type d -name "2*" | sort -r | head -n 1)
SVM_DIR=$(find logs/SVM -type d -name "2*" | sort -r | head -n 1)

# Update the feature selection config for SelectorMLP
echo "Step 4: Performing feature selection with SelectorMLP..."
TMP_CONFIG="configs/temp_feature_selection_mlp.json"
cat > $TMP_CONFIG << EOF
{
  "model_path": "${SELECTOR_DIR}",
  "adata_input": "intermediate_files/split_data/adata_val.h5ad",
  "y_column": "Main_cluster_name",
  "group_size": 50,
  "random_reps": 5,
  "results_dir": "results/feature_selection/selector_mlp"
}
EOF

# Run feature selection with SelectorMLP
python src/feature_selection.py --config $TMP_CONFIG
echo "SelectorMLP feature selection complete!"

# Update the feature selection config for SVM
echo "Step 5: Performing feature selection with SVM..."
TMP_CONFIG_SVM="configs/temp_feature_selection_svm.json"
cat > $TMP_CONFIG_SVM << EOF
{
  "model_path": "${SVM_DIR}",
  "adata_input": "intermediate_files/split_data/adata_val.h5ad",
  "y_column": "Main_cluster_name", 
  "group_size": 50,
  "random_reps": 5,
  "results_dir": "results/feature_selection/svm"
}
EOF

# Run feature selection with SVM
python src/feature_selection.py --config $TMP_CONFIG_SVM
echo "SVM feature selection complete!"

echo "Workflow completed successfully!"#