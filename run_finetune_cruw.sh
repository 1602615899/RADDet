export PYTHONPATH=$PYTHONPATH:/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node 4  \
    --master_port 29507  \
    main_finetune_cruw.py  \
    --batch_size 4   \
    --epochs 30 \
    --warmup_epochs 0 \
    --model cruw_tiny \
    --blr 1e-3 \
    --min_lr 1e-6 \
    --weight_decay 0.10 \
    --CRUW_config_path ./models/CRUW_finetune/config_Rodnet.py \
    --output_dir ./ft_CRUW_output_dir \
    --device cuda \
    --seed 42 \
    --distributed \
    --use_film_metadata \
    --bimamba_type none \
    --pin_mem \
    --num_workers 8 \
    --checkpoint_period 1\
    --data_type CRUW \
    --comment "cruw_tiny" \
    --finetune /media/ljm/Raid/ChenHongliang/RAGM/output_dir/best_model.pth \



    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250525_172940_mae_rpt_tiny_pretrain_ALL/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_CRUW_output_dir/20250523_115135_RADTR_cruw_tiny_finetune/checkpoint-30.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/output_dir/20250520_170730_mae_rpt_tiny_pretrain_CRUW/best_model.pth \
    # --finetune /mnt/truenas_users/ChenHongliang/RPT-master/ft_CRUW_output_dir/20250522_151043_RADTR_cruw_tiny_finetune/checkpoint-45.pth \
    # --dist_eval \
    # --define_loss \
    # --eval \
    # --epoch 45 \
    # --swanlab \


