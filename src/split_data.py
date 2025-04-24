from data import evenClusters
import argparse
import json
from anndata import read_h5ad
import os

def splitData(config):
    adata = read_h5ad(config["adata_path"])
    adata_train, adata_val, adata_test = evenClusters(
        adata, 
        config['y_column'],
        max_train=config['train_cluster_size'],
        max_val=config['val_cluster_size'], 
        max_test=config['test_cluster_size']
    )
    
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    adata_train.write_h5ad(os.path.join(save_dir, 'adata_train.h5ad'))
    adata_val.write_h5ad(os.path.join(save_dir, 'adata_val.h5ad'))
    adata_test.write_h5ad(os.path.join(save_dir, 'adata_test.h5ad'))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="configs/split_data.json")
    args = args.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    
    splitData(config)
        