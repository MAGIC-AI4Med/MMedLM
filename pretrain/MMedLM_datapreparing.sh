#!/bin/bash
#SBATCH -J cpu
#SBATCH -p CPU-64C256GB
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1      
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

srun python prepare_tokenized_dataset.py \
    --model_name_or_path 'internlm-chat-7b-v1_1' \
    --train_filenames 'TRAIN_FILENAMES_TXT' \
    --save_dir 'SAVE_DIR' \
    --cpu_num 16



