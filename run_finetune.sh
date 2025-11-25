#!/bin/bash

# GPU 设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

# 模型与训练参数
MODEL="raddet_tiny_bimamba_none"
BATCH_SIZE=12
EPOCHS=50
LR=1e-3
MIN_LR=1e-6
WEIGHT_DECAY=0.05
WARMUP=0
START_EPOCH=0
# 数据配置
DATA_TYPE="RADDet"
CONFIG_PATH="./models/RADDet_finetune/config.json"
OUTPUT_DIR="./ft_output_dir"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    exit 1
fi

# 训练命令
torchrun --nproc_per_node=4 --master_port=29505 main_finetune.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --blr $LR \
    --min_lr $MIN_LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP \
    --start_epoch $START_EPOCH \
    --data_type $DATA_TYPE \
    --RADDet_config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --distributed \
    --num_workers 8 \
    --pin_mem \
    --checkpoint_period 1 \
    --bias_wd \
    --clip_grad 1.0 \
    --comment "raddet_tiny_bimamba_none, 没有预训练, 加上FiLM,看看对性能影响有多大" \
    --swanlab \
    --use_film_metadata \
    --bimamba_type none \
    --seed 42 \
    --accum_iter 1 \
    # --finetune /media/ljm/Raid/ChenHongliang/RAGM/output_dir/20250919_010457_vision_mamba_3d_tiny_pretrain_ALL/best_model.pth \


    # --dist_eval \
    # --eval \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_output_dir/20250320_155251_RADTR_YOLO_tiny_finetune/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250317_094917_mae_rpt_tiny_pretrain_ALL/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250310_172229_mae_rpt_tiny_pretrain_ALL/best_model.pth \
    # --finetun    
    # --FPN \
