#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-4
#SBATCH --output=logs/domain-sampling-%j.out
#SBATCH --error=logs/domain-sampling-%j.err

cd $PROJECT_SCRATCH/mklimasz/CQF-side-effects

export APPTAINER_CACHEDIR=$PROJECT_SCRATCH/apptainer
srun singularity exec --nv -e \
    --bind $PROJECT_SCRATCH/mklimasz/CQF-side-effects/:/workspace,$HF_HOME:/workspace/cache \
    --pwd /workspace \
    --env "HF_HOME=/workspace/cache,HF_TOKEN=$HF_TOKEN" \
   docker://vllm/vllm-openai:v0.11.2 bash -c "pip install --user fire datasets accelerate \
   && python3 src/cqf/domain/sampling.py --input_path ./domain_scored --output_path ./domain_sampled"
