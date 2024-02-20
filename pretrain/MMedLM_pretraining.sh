#!/bin/bash
#SBATCH -J pretrain
#SBATCH -p GPU-8A100        
#SBATCH --nodes=1            
#SBATCH --ntasks=1           
#SBATCH --qos=gpu_8a100     
#SBATCH --gres=gpu:8       
#SBATCH --cpus-per-task=30
#SBATCH --mem=500G

torchrun --nproc_per_node=8 --master_port=29501 pretrain.py \
    --bf16 True \
    --output_dir OUTPUT_DIR \
    --train_dataset_path TRAIN_DATASET_FOLDER \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'InternLMDecoderLayer'\
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True\
    --model_max_length 2048 \
    --gradient_clipping 1.0
