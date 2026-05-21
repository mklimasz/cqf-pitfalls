#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=96GB
#SBATCH --gres=gpu:4
#SBATCH --time=0-8
#SBATCH --output=logs/generate-%j.out
#SBATCH --error=logs/generate-%j.err

cd $PROJECT_SCRATCH/mklimasz/CQF-side-effects
DOMAIN=$1

export APPTAINER_CACHEDIR=$PROJECT_SCRATCH/apptainer
srun singularity exec --nv -e \
    --bind $PROJECT_SCRATCH/mklimasz/CQF-side-effects/:/workspace,$HF_HOME:/workspace/cache \
    --pwd /workspace \
    --env "HF_HOME=/workspace/cache,HF_TOKEN=$HF_TOKEN" \
   docker://vllm/vllm-openai:v0.11.2 bash -c "pip install --user fire datasets \
   && python3 src/cqf/style/generation.py \
   --model_name_or_path Qwen/Qwen2.5-72B-Instruct \
   --output_path ./domain_generated/$DOMAIN.jsonl \
   --input_path ./domain_sampled/$DOMAIN.jsonl \
   --use_batch_writing False --subset_size 20000"
