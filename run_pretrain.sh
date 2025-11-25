OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node 4 \
    --master_port 29503  \
    -- \
    main_pretrain.py \
    --batch_size 8 \
    --epochs 200 \
    --warmup_epochs 0 \
    --start_epoch 0 \
    --model vision_mamba_3d_tiny \
    --blr 1e-3 \
    --min_lr 1e-6 \
    --weight_decay 0.10 \
    --autoreg_dim angle \
    --RADDet_config_path ./datasets/RADDet_config.json \
    --output_dir ./output_dir \
    --seed 42 \
    --pin_mem \
    --num_workers 8 \
    --checkpoint_period 1 \
    --data_type ALL \
    --swanlab \
    --accum_iter 4 \
    --comment "Pre-training vision_mamba_3d_tiny on ALL datasets" \
    # --resume /media/ljm/Raid/ChenHongliang/RAGM/output_dir/20250918_201051_vision_mamba_3d_tiny_pretrain_ALL/checkpoint-5.pth



    
    # --resume /mnt/SrvUserDisk/ZhangXu/pretrain/rpt/output_dir/20241109_212102_mae_rpt_base_pretrain/checkpoint-38.pth



