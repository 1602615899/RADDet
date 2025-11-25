OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node 4  \
    --master_port 29505  \
    main_finetune_carrada.py  \
    --batch_size 12   \
    --epochs 50 \
    --warmup_epochs 0 \
    --start_epoch 11 \
    --model carrada_tiny \
    --blr 1e-3 \
    --min_lr 1e-6 \
    --weight_decay 0.10 \
    --output_dir ./ft_Carrada_output_dir \
    --device cuda \
    --seed 42 \
    --distributed \
    --use_film_metadata \
    --bimamba_type none \
    --pin_mem \
    --num_workers 2 \
    --checkpoint_period 1\
    --data_type CARRADA \
    --comment "carrada_tiny" \
    --swanlab \
    --finetune /media/ljm/Raid/ChenHongliang/RAGM/ft_Carrada_output_dir/20250927_174609_carrada_tiny_finetune/checkpoint-11.pth
    # --dist_eval \
    # --define_loss \
    # --eval \
    # --eval \
    # --swanlab \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_Carrada_output_dir/20250519_010322_RADTR_YOLO_carrada_tiny_finetune/checkpoint-33.pth \

    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250514_140330_mae_rpt_tiny_8_pretrain_ALL/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_Carrada_output_dir/20250512_102354_RADTR_YOLO_carrada_tiny_finetune/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250410_105355_mae_rpt_tiny_pretrain_ALL/best_model.pth \

    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250317_094917_mae_rpt_tiny_pretrain_ALL/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250310_172229_mae_rpt_tiny_pretrain_ALL/best_model.pth \
    # --finetun    
    # --FPN \


# OMP_NUM_THREADS=2 \
# CUDA_VISIBLE_DEVICES=0,1 \
# torchrun \
#     --nproc_per_node 2  \
#     main_finetune.py  \
#     --batch_size 16 \
#     --epochs 50 \
#     --warmup_epochs 0 \
#     --model RADTR_YOLO_tiny \
#     --blr 1.6e-3 \
#     --min_lr 1e-6 \
#     --weight_decay 0.00 \
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
#     --resume ft_output_dir/20250108_214217_RADTR_YOLO_tiny_finetune/best_model.pth \
#     --all_mAP \