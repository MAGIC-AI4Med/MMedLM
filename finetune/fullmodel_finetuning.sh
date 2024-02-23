#!/bin/bash
#SBATCH -J fullmodel_fintuning
#SBATCH -p GPU-8A100          # 指定分区
#SBATCH --nodes=1            # 节点数
#SBATCH --ntasks=1           # 任务数
#SBATCH --qos=gpu_8a100      # 指定QoS
#SBATCH --gres=gpu:4       # 指定GPU资源
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24G

srun torchrun --nproc_per_node=4 --master_port 17999 train.py \
    --model_name_or_path MODEL_NAME_OR_PATH \
    --output_dir OUTPUT_DIR \
    --data_path "MMedBench/Train" \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap TRANSFORMER_LAYER \
    --gradient_checkpointing True\
    --gradient_clipping 1.0
