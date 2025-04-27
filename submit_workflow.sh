#!/bin/bash

sbatch --partition=hpc_a10_a \
       --time=4:00:00 \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=4 \
       --mem=128G \
       --gpus=a10:1 \
       --job-name=gene_selector_workflow \
       --output=workflow.out \
       --error=workflow.err \
       --wrap="source ~/.bashrc && conda activate torch && bash example_workflow.sh" 