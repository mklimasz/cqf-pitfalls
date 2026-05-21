#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/score-%j.out
#SBATCH --error=logs/score-%j.err

INPUT_PATH=$1
CLASSIFIER=$2
OUTPUT_PATH=$3

export APPTAINER_CACHEDIR=$PROJECT_SCRATCH/apptainer
srun singularity exec --nv -e \
    --bind $PROJECT_SCRATCH/mklimasz/CQF-side-effects/:/workspace,$HF_HOME:/workspace/cache \
    --pwd /workspace \
    --env "HF_HOME=/workspace/cache,HF_TOKEN=$HF_TOKEN" \
   docker://vllm/vllm-openai:v0.11.2 bash -c "python3 src/cqf/style/scoring.py \
   --input_path $INPUT_PATH \
   --batch_size 64 \
   --output_path $OUTPUT_PATH \
   --cqf_model_name_or_path $CLASSIFIER"
