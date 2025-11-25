import math
import os
import sys
from typing import Iterable
import swanlab
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.CARRADA_finetune.utils.functions import get_metrics, get_qualitatives
from models.CARRADA_finetune.utils.metrics import Evaluator
import utils
import torch.distributed as dist
import time
# 引入 matplotlib.pyplot 库用于保存图片和绘制
import matplotlib.pyplot as plt
import matplotlib # 引入 matplotlib 用于获取 Colormap 等
import matplotlib.colors as mcolors # 用于创建自定义 Colormap
from torch.amp import autocast

def save_grayscale_img_mpl_rad(numpy_array, filepath, title=None):
    # 确保数组是 2D
    if numpy_array.ndim != 2:
        print(f"Warning: Array {filepath} is not 2D ({numpy_array.ndim}D), skipping image save.")
        return

    # 处理潜在的 NaN 值
    numpy_array = np.nan_to_num(numpy_array)

    # 创建一个图窗和坐标轴
    fig, ax = plt.subplots(figsize=(4,4), dpi=300) # 根据图片尺寸设置 figsize 和 dpi
    # 使用 imshow 显示灰度图，让 Matplotlib 自动处理颜色缩放
    im = ax.imshow(numpy_array, cmap='gray')

    # 根据 title 添加自定义的坐标轴标签和刻度，仿照 imgPlot 函数
    if title == "RD":
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_xticklabels([-13, -6.5, 0, 6.5, 13])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("velocity (m/s)")
        ax.set_ylabel("range (m)")
    elif title == "RA":
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("range (m)")
    # 如果还有 DA 视图需要保存，可以在这里添加对应的标签设置
    # elif title == "DA":
    #     ax.set_xticks([...])
    #     ax.set_xticklabels([...])
    #     ax.set_yticks([...])
    #     ax.set_yticklabels([...])
    #     ax.set_xlabel(...)
    #     ax.set_ylabel(...)

    if title is not None:
        ax.set_title(title)

    # 可以选择添加 colorbar
    # fig.colorbar(im, ax=ax)

    # 使用 tight_layout 保存，并调整边距
    plt.tight_layout(pad=0.1)
    plt.savefig(filepath, bbox_inches='tight') # bbox_inches='tight' 尝试去除白边
    plt.close(fig) # 关闭图窗以释放内存

def train_one_epoch(
    model: torch.nn.Module,
    model_ema,
    rd_criterion: torch.nn.Module,
    ra_criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    ):

    if len(data_loader) == 0:
        raise ValueError("DataLoader is empty! Check dataset paths and batch size.")
    
    print('RD Criterion:', rd_criterion)
    print('RA Criterion:', ra_criterion)

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # add metrics
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("total_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("rd_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("ra_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("coherence_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    print_freq = 20     # 每20个batch打印一次

    accum_iter = args.accum_iter if hasattr(args, 'accum_iter') else 1

    running_stats = {
        'total_loss': [],
        'rd_loss': [],
        'ra_loss': [],
        'coherence_loss': []
    }

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # if utils.is_main_process():
        #     print(f"--- Visualizing first batch of epoch {epoch} for validation data ---")
        #     visualize_batch(data, args.output_dir, epoch, batch_idx=data_iter_step, prefix='val')

        if data_iter_step % accum_iter == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        rd_mask = data['rd_mask'].to(device).float()
        ra_mask = data['ra_mask'].to(device).float() 
        samples = data['rad'].to(device, non_blocking=True)
        condition = data['condition'].to(device, non_blocking=True)
        raw_params = {k: v.to(device, non_blocking=True) for k, v in data['raw_params'].items()}

        optimizer.zero_grad()        
        with autocast(device_type='cuda', dtype=torch.float16):
            rd_outputs, ra_outputs = model(samples, condition=condition, batch_params=raw_params)

            rd_losses_list = [c(rd_outputs, torch.argmax(rd_mask, axis=1)) for c in rd_criterion[:2]]
            rd_loss = torch.mean(torch.stack(rd_losses_list))

            ra_losses_list = [c(ra_outputs, torch.argmax(ra_mask, axis=1)) for c in ra_criterion[:2]]
            ra_loss = torch.mean(torch.stack(ra_losses_list))

            coherence_loss = rd_criterion[2](rd_outputs, ra_outputs) # 假设此 criterion 也与 AMP 兼容
            total_loss = rd_loss + ra_loss + coherence_loss

        running_stats['total_loss'].append(total_loss.item())
        running_stats['rd_loss'].append(rd_loss.item())
        running_stats['ra_loss'].append(ra_loss.item())
        running_stats['coherence_loss'].append(coherence_loss.item())
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)
        total_loss /= accum_iter

        
        loss_scaler(
            total_loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            # clip_grad=args.clip_grad,   # 梯度裁剪
        )

        if (data_iter_step + 1) % accum_iter == 0:
            model_ema.update() 
            optimizer.zero_grad()

        torch.cuda.synchronize()    # 同步cuda 等待所有在 GPU 上进行的操作完成
        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"],
            total_loss=total_loss.item(),
            rd_loss=rd_loss.item(),
            ra_loss=ra_loss.item(),
            coherence_loss=coherence_loss.item()
        )

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)
            log_writer.add_scalar('loss/total', total_loss.item(), epoch_1000x)
            log_writer.add_scalar('loss/rd', rd_loss.item(), epoch_1000x)
            log_writer.add_scalar('loss/ra', ra_loss.item(), epoch_1000x)
            log_writer.add_scalar('loss/coherence', coherence_loss.item(), epoch_1000x)


        if args.swanlab and utils.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
            # 计算全局步数，确保图表是连续的
            global_step = epoch * len(data_loader) + data_iter_step
            swanlab.log({
                'train/learning_rate': optimizer.param_groups[0]["lr"],
                'train/total_loss': total_loss.item(),
                'train/rd_loss': rd_loss.item(),
                'train/ra_loss': ra_loss.item(),
                'train/coherence_loss': coherence_loss.item(),
            }, step=global_step)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("训练统计:")
    print(f"平均总损失: {np.mean(running_stats['total_loss']):.4f}")
    print(f"平均RD损失: {np.mean(running_stats['rd_loss']):.4f}")
    print(f"平均RA损失: {np.mean(running_stats['ra_loss']):.4f}")
    print(f"平均一致性损失: {np.mean(running_stats['coherence_loss']):.4f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(    
    model: torch.nn.Module,
    rd_criterion: torch.nn.Module,
    ra_criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int, # <-- 添加 epoch 参数，用于命名
    args=None,
    ):

    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    
    metric_logger.add_meter("total_loss_val", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("rd_loss_val", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("ra_loss_val", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("coherence_loss_val", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    rd_metrics = Evaluator(num_class=4)
    ra_metrics = Evaluator(num_class=4)
    rd_metrics.reset() # Ensure clean state
    ra_metrics.reset()

    # switch to evaluation mode
    model.eval()
    running_losses = list()
    rd_running_losses = list()
    rd_running_global_losses = [list(), list()]
    ra_running_losses = list()
    ra_running_global_losses = [list(), list()]
    coherence_running_losses = list()    
    print_freq = 10
    with torch.no_grad():
        for data_iter_step, data in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
            rd_mask = data['rd_mask'].to(device).float()
            ra_mask = data['ra_mask'].to(device).float()        
            samples = data['rad'].to(device, non_blocking=True)
            condition = data['condition'].to(device, non_blocking=True)
            raw_params = {k: v.to(device, non_blocking=True) for k, v in data['raw_params'].items()}
                
            rd_outputs, ra_outputs = model(samples, condition=condition, batch_params=raw_params)

            rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                torch.argmax(rd_outputs, axis=1).cpu())
            ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                torch.argmax(ra_outputs, axis=1).cpu())
            total_loss_val, rd_loss_val, ra_loss_val, coherence_loss_val = torch.tensor(0.0, device=device), \
                                                                    torch.tensor(0.0, device=device), \
                                                                    torch.tensor(0.0, device=device), \
                                                                    torch.tensor(0.0, device=device)
            
            # box_loss_b, conf_loss_b, category_loss_b = box_loss.item(), conf_loss.item(), category_loss.item()
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1)) for c in rd_criterion[:2]]
            rd_loss_val = torch.mean(torch.stack(rd_losses))

            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1)) for c in ra_criterion[:2]]
            ra_loss_val = torch.mean(torch.stack(ra_losses))

            coherence_loss_val = rd_criterion[2](rd_outputs, ra_outputs)
            total_loss_val = torch.mean(rd_loss_val + ra_loss_val + coherence_loss_val)

            running_losses.append(total_loss_val.item())
            rd_running_losses.append(rd_loss_val.item())
            ra_running_losses.append(ra_loss_val.item())
            coherence_running_losses.append(coherence_loss_val.item())

            metric_logger.update(
                total_loss_val=total_loss_val.item(),
                rd_loss_val=rd_loss_val.item(),
                ra_loss_val=ra_loss_val.item(),
                coherence_loss_val=coherence_loss_val.item()
            )

    test_results = dict()
    test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses))
    test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses))
    test_results['coherence_loss'] = np.mean(coherence_running_losses).item()
    test_results['global_acc'] = (1/2)*(test_results['range_doppler']['acc']+
                                            test_results['range_angle']['acc'])
    test_results['global_prec'] = (1/2)*(test_results['range_doppler']['prec']+
                                            test_results['range_angle']['prec'])
    test_results['global_dice'] = (1/2)*(test_results['range_doppler']['dice']+
                                            test_results['range_angle']['dice'])

    rd_metrics.reset()
    ra_metrics.reset()

    return test_results

# 将这个函数添加到你的 engine.py 文件的顶部

def visualize_batch(batch_data, save_dir, epoch, batch_idx=0, prefix=''):
    """
    可视化一个批次中的数据、GT，并保存为图像。
    
    Args:
        batch_data (dict): 从 DataLoader 中获取的数据字典。
        save_dir (str): 保存图像的根目录。
        epoch (int): 当前的 epoch 数。
        batch_idx (int): 当前的 batch 索引。
        prefix (str): 文件名前缀，如 'train' 或 'val'。
    """
    # 确保保存目录存在
    vis_path = os.path.join(save_dir, "visualizations", f"epoch_{epoch:03d}")
    os.makedirs(vis_path, exist_ok=True)
    
    # 从批次中解包数据 (移动到 CPU 并转为 numpy)
    samples = batch_data['rad'].cpu().numpy()
    rd_mask = batch_data['rd_mask'].cpu().numpy()
    ra_mask = batch_data['ra_mask'].cpu().numpy()
    
    batch_size = samples.shape[0]


    for i in range(batch_size):
        # --- 1. 可视化输入 RAD 数据 ---
        rad_sample = samples[i, 0] # (D, H, W) -> (Doppler, Azimuth, Range)
        
        # 提取 RD 视图 (在 Azimuth 维度上取平均)
        rd_view = np.mean(rad_sample, axis=1) # (D, W)
        
        # 提取 RA 视图 (在 Doppler 维度上取平均)
        ra_view = np.mean(rad_sample, axis=-1) # (H, W)
        
        # 保存 RD 视图
        save_grayscale_img_mpl_rad(
            rd_view, # 转置以匹配 (Range, Doppler)
            os.path.join(vis_path, f"{prefix}_batch{batch_idx}_sample{i}_input_RD.png"),
            title="RD"
        )
        
        # 保存 RA 视图
        save_grayscale_img_mpl_rad(
            ra_view, # 转置以匹配 (Range, Azimuth)
            os.path.join(vis_path, f"{prefix}_batch{batch_idx}_sample{i}_input_RA.png"),
            title="RA"
        )
        
        # --- 2. 可视化真值标签 (Ground Truth) ---
        # 将 one-hot 编码的 mask 转换为类别图
        rd_gt = np.argmax(rd_mask[i], axis=0) # (R, D)
        ra_gt = np.argmax(ra_mask[i], axis=0) # (R, A)
        
        # 使用自定义的颜色映射来可视化 GT
        # 0: 背景 (黑), 1: Car (红), 2: Pedestrian (绿), 3: Cyclist (蓝)
        colors = ['black', 'red', 'green', 'blue']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
        
        # 保存 RD GT
        plt.imsave(
            os.path.join(vis_path, f"{prefix}_batch{batch_idx}_sample{i}_gt_RD.png"),
            rd_gt,
            cmap=cmap,
            vmin=0,
            vmax=3
        )
        
        # 保存 RA GT
        plt.imsave(
            os.path.join(vis_path, f"{prefix}_batch{batch_idx}_sample{i}_gt_RA.png"),
            ra_gt,
            cmap=cmap,
            vmin=0,
            vmax=3
        )
    