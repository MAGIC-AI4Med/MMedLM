#!/bin/bash
#SBATCH -J inference
#SBATCH -p GPU-8A100
#SBATCH --nodes=1 
#SBATCH --ntasks=1    
#SBATCH --qos=gpu_8a100  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G

srun python inference.py \
    --model_name_or_path MODEL_NAME_OR_PATH \
    --is_lora IS_LORA \
    --save_dir OUTPUT_DIR \
    --is_with_rationale IS_WITH_RATIONALE \
    --data_path "MMedBench/Test"
