#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=96GB
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/generate-%j.out
#SBATCH --error=logs/generate-%j.err

cd $PROJECT_SCRATCH/mklimasz/CQF-side-effects

export APPTAINER_CACHEDIR=$PROJECT_SCRATCH/apptainer
srun singularity exec --nv -e \
    --bind $PROJECT_SCRATCH/mklimasz/CQF-side-effects/:/workspace,$HF_HOME:/workspace/cache \
    --pwd /workspace \
    --env "HF_HOME=/workspace/cache,HF_TOKEN=$HF_TOKEN" \
   docker://vllm/vllm-openai:v0.11.2 bash -c "python3 src/cqf/edu/labelling.py ./scored/fineweb-edu-classifier ./labelling/fineweb-edu-classifier meta-llama/Llama-3.1-70B-Instruct"
