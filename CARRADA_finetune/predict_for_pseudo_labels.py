# predict_for_pseudo_labels.py

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging
import sys
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# Assuming your project structure allows these imports
from models.RADDet_finetune import loader
import utils 
from utils import NativeScalerWithGradNormCount as NativeScaler # May not be needed if not training
from models.CARRADA_finetune.CARRADA_finetune_dataset import Create_CARRADA_Finetune_Dataset # Or a new dataset class for unlabeled data
import models.radtr_carrada as models_radtr_yolo_carrada
# from torch_ema import ExponentialMovingAverage # If you plan to load EMA models

# --- Argument Parser (adapted from main_finetune_carrada.py) ---
def get_args_parser():
    parser = argparse.ArgumentParser('Pseudo-label generation script', add_help=False)
    
    # --- Essential parameters ---
    parser.add_argument('--model', default='RADTR_YOLO_carrada_tiny', type=str, metavar='MODEL',
                        help='Name of model to use for prediction')
    parser.add_argument('--checkpoint_path', default='/mnt/truenas_users/ChenHongliang/RPT-master/ft_Carrada_output_dir/20250519_010322_RADTR_YOLO_carrada_tiny_finetune/checkpoint-33.pth', type=str,
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--output_dir', default='/mnt/truenas_users/ChenHongliang/RPT-master/label0.0', type=str,
                        help='Directory to save pseudo-labels and/or probability maps')
    parser.add_argument('--CARRADA_config_path', default='./models/CARRADA_finetune/CARRADA.json', type=str,
                        help='Path to CARRADA.json for dataset/model configurations if needed')
    
    # --- Prediction control ---
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Confidence threshold for selecting pseudo-labels (0.0 to 1.0)')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Also save the raw probability maps')
    parser.add_argument('--ignore_index', type=int, default=255,
                        help='Value to use for pixels below confidence threshold in pseudo-label masks')

    # --- Standard parameters (from main_finetune_carrada.py, some may be optional for prediction) ---
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--device', default='cuda:3', help='Device to use for prediction (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Model specific (if needed by radtr_yolo_carrada model loader)
    parser.add_argument('--attn_type', default="Normal", type=str, help="Attention type for the model")
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop path rate in model')
    
    # Distributed (optional, if you want to run prediction in DDP)
    parser.add_argument("--distributed", action="store_true", default=False, help="Enable distributed prediction")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

# --- Main Prediction Loop (adapted from engine_finetune_carrada.py's evaluate) ---
@torch.no_grad()
def generate_pseudo_labels(args, model, data_loader, device):
    model.eval()
    
    # ... (创建输出目录的代码不变) ...
    output_pseudo_labels_rd_dir = Path(args.output_dir) / "pseudo_labels_rd"
    output_pseudo_labels_ra_dir = Path(args.output_dir) / "pseudo_labels_ra"
    output_pseudo_labels_rd_dir.mkdir(parents=True, exist_ok=True)
    output_pseudo_labels_ra_dir.mkdir(parents=True, exist_ok=True)

    if args.save_probabilities:
        output_probs_rd_dir = Path(args.output_dir) / "probs_rd"
        output_probs_ra_dir = Path(args.output_dir) / "probs_ra"
        output_probs_rd_dir.mkdir(parents=True, exist_ok=True)
        output_probs_ra_dir.mkdir(parents=True, exist_ok=True)

    # ... (TTA配置和 metric_logger 初始化不变) ...
    augmentations_config = [
        {'name': 'RangeFlip', 'input_flip_dims': [3], 'output_rd_flip_dims': [2], 'output_ra_flip_dims': [2]},
        {'name': 'AngleFlip', 'input_flip_dims': [4], 'output_rd_flip_dims': [], 'output_ra_flip_dims': [3]},
        {'name': 'DopplerFlip', 'input_flip_dims': [2], 'output_rd_flip_dims': [3], 'output_ra_flip_dims': []}
    ]
    print_freq = 20
    header = "Generating Pseudo-Labels:"
    metric_logger = utils.MetricLogger(delimiter="  ") if hasattr(utils, 'MetricLogger') else None
    
    iterable_data_loader = data_loader
    if metric_logger:
        iterable_data_loader = metric_logger.log_every(data_loader, print_freq, header)
    
    num_classes = 4 # CARRADA 类别数 (例如 背景, 行人, 汽车, 自行车)

    for batch_idx, data_batch in enumerate(iterable_data_loader):
        samples_raw = data_batch['rad'].to(device, non_blocking=True)
        sample_ids_in_batch = data_batch.get('sample_id_str', []) # 确保与Dataset返回的键名一致

        samples_model_input_orig = samples_raw.permute(0, 3, 1, 2).unsqueeze(1)
        batch_size = samples_model_input_orig.shape[0]
        
        tta_rd_logits_list = []
        tta_ra_logits_list = []

        # 1. 原始预测
        rd_logits_orig, ra_logits_orig = model(samples_model_input_orig)
        tta_rd_logits_list.append(rd_logits_orig)
        tta_ra_logits_list.append(ra_logits_orig)

        # 2. TTA 预测
        for aug_params in augmentations_config:
            augmented_input = torch.flip(samples_model_input_orig, dims=aug_params['input_flip_dims'])
            rd_logits_aug, ra_logits_aug = model(augmented_input)
            if aug_params['output_rd_flip_dims']:
                rd_logits_aug = torch.flip(rd_logits_aug, dims=aug_params['output_rd_flip_dims'])
            if aug_params['output_ra_flip_dims']:
                ra_logits_aug = torch.flip(ra_logits_aug, dims=aug_params['output_ra_flip_dims'])
            tta_rd_logits_list.append(rd_logits_aug)
            tta_ra_logits_list.append(ra_logits_aug)

        # 3. 聚合TTA结果 (平均logits)
        final_rd_logits = torch.mean(torch.stack(tta_rd_logits_list), dim=0)
        final_ra_logits = torch.mean(torch.stack(tta_ra_logits_list), dim=0)
        
        final_rd_probs = F.softmax(final_rd_logits, dim=1) # (B, 4, R_out, D_out)
        final_ra_probs = F.softmax(final_ra_logits, dim=1) # (B, 4, R_out, A_out)
        
        # --- 生成高置信度类别索引伪标签 (形状 B, H, W) ---
        max_rd_probs, pseudo_labels_rd_indices = torch.max(final_rd_probs, dim=1)
        high_confidence_rd_mask = max_rd_probs >= args.confidence_threshold
        output_pseudo_labels_rd_indices = torch.full_like(pseudo_labels_rd_indices, args.ignore_index, dtype=torch.long)
        output_pseudo_labels_rd_indices[high_confidence_rd_mask] = pseudo_labels_rd_indices[high_confidence_rd_mask]

        max_ra_probs, pseudo_labels_ra_indices = torch.max(final_ra_probs, dim=1)
        high_confidence_ra_mask = max_ra_probs >= args.confidence_threshold
        output_pseudo_labels_ra_indices = torch.full_like(pseudo_labels_ra_indices, args.ignore_index, dtype=torch.long)
        output_pseudo_labels_ra_indices[high_confidence_ra_mask] = pseudo_labels_ra_indices[high_confidence_ra_mask]

        # --- 将索引图伪标签转换为 one-hot 格式 (B, NumClasses, H, W) ---
        # 对于RD视图
        temp_rd_indices_for_onehot = output_pseudo_labels_rd_indices.clone()
        # 将 ignore_index 的像素临时标记为0 (或任何有效类别索引)，以便 F.one_hot 不会出错
        # 稍后我们会将这些位置的one-hot向量清零
        valid_pixels_rd_mask = (temp_rd_indices_for_onehot != args.ignore_index)
        temp_rd_indices_for_onehot[~valid_pixels_rd_mask] = 0 
        
        # 执行one-hot转换，得到 (B, H, W, NumClasses)
        pseudo_labels_rd_one_hot_bhwc = F.one_hot(temp_rd_indices_for_onehot, num_classes=num_classes)
        
        # 将原先是 ignore_index 的像素位置的 one-hot 向量设置为全零
        # valid_pixels_rd_mask 需要扩展维度以进行广播赋值
        pseudo_labels_rd_one_hot_bhwc[~valid_pixels_rd_mask] = 0 # 设置为全零向量 [0,0,0,0]
        
        # 转换维度为 (B, NumClasses, H, W) 并确保是 float 类型
        pseudo_labels_rd_one_hot_bchw = pseudo_labels_rd_one_hot_bhwc.permute(0, 3, 1, 2).float()

        # 对于RA视图 (类似操作)
        temp_ra_indices_for_onehot = output_pseudo_labels_ra_indices.clone()
        valid_pixels_ra_mask = (temp_ra_indices_for_onehot != args.ignore_index)
        temp_ra_indices_for_onehot[~valid_pixels_ra_mask] = 0
        pseudo_labels_ra_one_hot_bhwc = F.one_hot(temp_ra_indices_for_onehot, num_classes=num_classes)
        pseudo_labels_ra_one_hot_bhwc[~valid_pixels_ra_mask] = 0
        pseudo_labels_ra_one_hot_bchw = pseudo_labels_ra_one_hot_bhwc.permute(0, 3, 1, 2).float()
        # --- one-hot 转换完毕 ---

        for i in range(batch_size):
            sample_id_str = "unknown_sample" 
            if i < len(sample_ids_in_batch) and sample_ids_in_batch[i] is not None:
                sample_id_str = str(sample_ids_in_batch[i])
            else: 
                base_name = f"batch{batch_idx}_idx{i}"
                if utils.is_dist_avail_and_initialized() and hasattr(utils, 'get_rank'):
                    sample_id_str = f"rank{utils.get_rank()}_{base_name}"
                else:
                    sample_id_str = base_name
            
            # --- 保存 one-hot 格式的伪标签 ---
            # RD伪标签 (4, R_out, D_out)
            pseudo_label_rd_to_save_np = pseudo_labels_rd_one_hot_bchw[i].cpu().numpy() # 已经是 (C,H,W)
            # 文件名可以指明是onehot格式，例如
            rd_label_filename = f"{sample_id_str}_rd_pseudo_label_onehot.npy" 
            rd_label_path = output_pseudo_labels_rd_dir / rd_label_filename
            np.save(rd_label_path, pseudo_label_rd_to_save_np.astype(np.float32)) # 保存为float32

            # RA伪标签 (4, R_out, A_out)
            pseudo_label_ra_to_save_np = pseudo_labels_ra_one_hot_bchw[i].cpu().numpy()
            ra_label_filename = f"{sample_id_str}_ra_pseudo_label_onehot.npy"
            ra_label_path = output_pseudo_labels_ra_dir / ra_label_filename
            np.save(ra_label_path, pseudo_label_ra_to_save_np.astype(np.float32))
            # ---

            if args.save_probabilities:
                # 保存概率图的逻辑不变
                probs_rd_np = final_rd_probs[i].cpu().numpy().astype(np.float16)
                rd_probs_filename = f"{sample_id_str}_rd_probs.npy"
                rd_probs_path = output_probs_rd_dir / rd_probs_filename
                np.save(rd_probs_path, probs_rd_np)

                probs_ra_np = final_ra_probs[i].cpu().numpy().astype(np.float16)
                ra_probs_filename = f"{sample_id_str}_ra_probs.npy"
                ra_probs_path = output_probs_ra_dir / ra_probs_filename
                np.save(ra_probs_path, probs_ra_np)
                
    print(f"Pseudo-label generation complete. Outputs saved to: {args.output_dir}")


# --- Main execution logic (adapted from main_finetune_carrada.py) ---
def main():
    args = get_args_parser().parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        device = torch.device(args.gpu)
    else:
        device = torch.device(args.device)
    
    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    _, dataset_val = Create_CARRADA_Finetune_Dataset('Test')

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, drop_last=False, shuffle=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) # Samples elements sequentially, always in the same order.
    else:
        num_tasks = 1
        global_rank = 0
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    print(len(dataset_val))
    data_loader_unlabeled = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存

    )    

    # --- Model Definition ---
    print(f"Creating model: {args.model}")
    # Ensure your model factory can take n_classes if it's part of its signature,
    # otherwise, it might be hardcoded in the model class itself.
    # config = models.RADDet_finetune.loader.readConfig(config_file_name=args.CARRADA_config_path) # If needed for n_classes
    # num_actual_classes = config.get('nb_classes', 4)

    model = models_radtr_yolo_carrada.__dict__[args.model](
        attn_type=args.attn_type,
        drop_path_rate=args.drop_path_rate,
        # n_classes=num_actual_classes # Pass if your model constructor takes it
    )
    
    # --- Load Checkpoint ---
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

    print("Load pre-trained checkpoint from: %s" % args.checkpoint_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
                        
    # interpolate position embedding
    utils.interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model    strict=False 允许权重部分匹配，decoder部分不会被加载
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu],    # device_ids=[torch.cuda.current_device()]
                                                          find_unused_parameters=False       # change True to False
                                                          )
        model_without_ddp = model.module


    # --- Start Prediction ---
    print("Starting pseudo-label generation process...")
    generate_pseudo_labels(args, model, data_loader_unlabeled, device)

    print("Pseudo-label generation finished.")

if __name__ == '__main__':
    main()