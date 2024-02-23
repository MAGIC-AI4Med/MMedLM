#!/bin/bash
#SBATCH -J LoRA_fintuning
#SBATCH -p GPU-8A100          # 指定分区
#SBATCH --nodes=1            # 节点数
#SBATCH --ntasks=1           # 任务数
#SBATCH --qos=gpu_8a100      # 指定QoS
#SBATCH --gres=gpu:4       # 指定GPU资源
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24G

srun torchrun --nproc_per_node=4 --master_port 17998 train.py \
    --model_name_or_path MODEL_NAME_OR_PATH \
    --data_path "MMedBench/Train" \
    --output_dir OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --is_lora True\
    --local_rank 16\
    --target_modules TARGET_MODUILES
