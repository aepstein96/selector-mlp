#!/bin/bash
# Script to submit the pipeline as an SLURM batch job

# Set default values
INPUT_DATA="raw_data/adata_variable_small.h5ad"
SPLIT_DIR="intermediate_files/split_datasets"
USE_SMALL_SPLIT=false
RESUME_TRAINING=false
RUN_DATA_SPLIT=true
MAX_EPOCHS=50
JOB_NAME="gene_selector"
USE_GPU=false
PARTITION="hpc"
CPU_COUNT=16
MEMORY="64G"
TIME="24:00:00"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --input)
            INPUT_DATA="$2"
            shift 2
            ;;
        --split_dir)
            SPLIT_DIR="$2"
            shift 2
            ;;
        --small_split)
            USE_SMALL_SPLIT=true
            shift
            ;;
        --resume)
            RESUME_TRAINING=true
            shift
            ;;
        --no_split)
            RUN_DATA_SPLIT=false
            shift
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --cpus)
            CPU_COUNT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p slurm_logs

# Prepare the pipeline command
PIPELINE_CMD="./run_pipeline.sh"
if [ "$INPUT_DATA" != "raw_data/adata_variable_small.h5ad" ]; then
    PIPELINE_CMD+=" --input ${INPUT_DATA}"
fi
if [ "$SPLIT_DIR" != "intermediate_files/split_datasets" ]; then
    PIPELINE_CMD+=" --split_dir ${SPLIT_DIR}"
fi
if [ "$USE_SMALL_SPLIT" = true ]; then
    PIPELINE_CMD+=" --small_split"
fi
if [ "$RESUME_TRAINING" = true ]; then
    PIPELINE_CMD+=" --resume"
fi
if [ "$RUN_DATA_SPLIT" = false ]; then
    PIPELINE_CMD+=" --no_split"
fi
if [ "$MAX_EPOCHS" != "50" ]; then
    PIPELINE_CMD+=" --epochs ${MAX_EPOCHS}"
fi

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="slurm_logs/${JOB_NAME}_${TIMESTAMP}"

# Prepare SLURM submission script
if [ "$USE_GPU" = true ]; then
    # GPU job configuration (login05/Rocky 9)
    sbatch_cmd="sbatch \
        --partition=hpc_a10_a \
        --time=${TIME} \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=${CPU_COUNT} \
        --mem=${MEMORY} \
        --gpus=a10:1 \
        --job-name=${JOB_NAME} \
        --output=${LOG_FILE}.out \
        --error=${LOG_FILE}.err \
        --wrap=\"source ~/.bashrc && conda activate torch && cd $(pwd) && chmod +x run_pipeline.sh && ${PIPELINE_CMD}\""
else
    # CPU job configuration
    sbatch_cmd="sbatch \
        --partition=${PARTITION} \
        --time=${TIME} \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=${CPU_COUNT} \
        --mem=${MEMORY} \
        --job-name=${JOB_NAME} \
        --output=${LOG_FILE}.out \
        --error=${LOG_FILE}.err \
        --wrap=\"source ~/.bashrc && conda activate torch && cd $(pwd) && chmod +x run_pipeline.sh && ${PIPELINE_CMD}\""
fi

echo "Submitting job with the following command:"
echo "$sbatch_cmd"
eval "$sbatch_cmd"

echo "Job submitted. Check status with 'squeue -u $USER'"
echo "Log files will be written to ${LOG_FILE}.out and ${LOG_FILE}.err" 