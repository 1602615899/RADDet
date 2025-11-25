OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node 4 \
    --master_port 29502 \
    main_finetune_cart.py \
    --batch_size 8 \
    --epochs 50 \
    --warmup_epochs 0 \
    --model RADTR_YOLO_tiny_cart \
    --blr 1e-3 \
    --min_lr 1e-6 \
    --weight_decay 0.10 \
    --drop_path_rate 0.00 \
    --data_path not_used \
    --RADDet_config_path ./models/RADDet_finetune/config_cart.json \
    --output_dir ./ft_output_dir_cart \
    --device cuda:0 \
    --seed 42 \
    --pin_mem \
    --num_workers 2 \
    --attn_type Normal \
    --checkpoint_period 1 \
    --data_type RADDet \
    --comment "二维检测" \
    --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_output_dir_cart/best_model.pth

    #  --distributed \
    #  --dist_eval \

    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_output_dir/20250317_095056_RADTR_YOLO_tiny_finetune/checkpoint-21.pth \

    # --swanlab  \

    # --master_port 29501  \


# OMP_NUM_THREADS=2 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#     --nproc_per_node 4  \
#     main_finetune.py  \
#     --batch_size 20 \
#     --epochs 50 \
#     --warmup_epochs 0 \
#     --model RADTR_YOLO_tiny \
#     --blr 1.6e-3 \
#     --min_lr 1e-6 \
#     --weight_decay 0.10 \
#     --drop_path_rate 0.00 \
#     --data_path not_used \
#     --RADDet_config_path ./models/RADDet_finetune/config.json \
#     --output_dir ./ft_output_dir \
#     --device cuda:0 \
#     --seed 42 \
#     --pin_mem \
#     --num_workers 2 \
#     --distributed \
#     --attn_type Normal \
#     --checkpoint_period 10\
#     --data_type RADDet \
#     --comment "ViTDet, val, dist eval, not shuffle" \
#     --eval \
#     --dist_eval \
#     --resume ft_output_dir/20241228_200739_RADTR_YOLO_tiny_finetune/checkpoint-60.pth \