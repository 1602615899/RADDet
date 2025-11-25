# engine_pretrain.py - 最终版本
import math
import sys
from typing import Iterable
import torch
import utils
import swanlab
import numpy as np

# engine_pretrain.py 中的修改部分
def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    print_freq = 20     # 每20个batch打印一次

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 修改：安全地处理数据加载器的返回值
    for data_iter_step, batch_item in enumerate(              
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = batch_item['data'].to(device, non_blocking=True)
        condition = batch_item['condition'].to(device, non_blocking=True)
        raw_params = {k: v.to(device, non_blocking=True) for k, v in batch_item['raw_params'].items()}

        
        loss_tensor = model(samples, condition=condition, batch_params=raw_params)
        loss = loss_tensor.mean()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"!!!!!!!!!!!!!! PROBLEM DETECTED after loss.mean() !!!!!!!!!!!!!!")
            print(f"Epoch: {epoch}, Step: {data_iter_step}")
            print(f"  Aggregated loss_value is: {loss_value}")
            
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if args.swanlab and utils.is_main_process():
            current_step = epoch * len(data_loader) + data_iter_step
            swanlab.log({
                'batch_loss': loss_value,       # 当前批次的损失
                'batch_lr': lr,                 # 当前批次的学习率
                'epoch': epoch,
                'step': current_step            # 全局步数
            }, step=current_step) 

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}