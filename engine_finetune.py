import math
import sys
from typing import Iterable
import swanlab
import torch
import numpy as np

import utils
from models.RADDet_finetune.yolo_head import decodeYolo, decodeYolo_fpn, yoloheadToPredictions, nms
import models.RADDet_finetune.mAP as mAP
import torch.distributed as dist
import time



def process_batch_params(samples_shape, batch_dataset_ids=None, batch_params=None, batch_dataset_names=None):
    """处理批次参数"""
    batch_size = samples_shape[0]
    
    # 处理数据集ID
    if batch_dataset_ids is not None:
        if isinstance(batch_dataset_ids, (int, float)):
            batch_dataset_ids = torch.full((batch_size,), int(batch_dataset_ids), dtype=torch.long)
        elif isinstance(batch_dataset_ids, (list, tuple)):
            batch_dataset_ids = torch.tensor(batch_dataset_ids, dtype=torch.long)
        elif isinstance(batch_dataset_ids, torch.Tensor):
            if batch_dataset_ids.dim() == 0:
                batch_dataset_ids = batch_dataset_ids.unsqueeze(0).repeat(batch_size)
        if isinstance(batch_dataset_ids, torch.Tensor):
            batch_dataset_ids = batch_dataset_ids.long()
    else:
        batch_dataset_ids = None
    
    # 处理数据集参数
    if batch_params is not None:
        if isinstance(batch_params, dict):
            batch_params = [batch_params] * batch_size
        elif not isinstance(batch_params, (list, tuple)):
            # 默认参数 (RADDet参数)
            batch_params = [{
                'range_res': 0.1953125,
                'angle_res': np.deg2rad(0.67),
                'vel_res': 0.41968,
                'has_velocity': True
            }] * batch_size
    else:
        batch_params = None
    
    # 处理数据集名称
    if batch_dataset_names is not None:
        if isinstance(batch_dataset_names, str):
            batch_dataset_names = [batch_dataset_names] * batch_size
        elif not isinstance(batch_dataset_names, (list, tuple)):
            batch_dataset_names = ['Unknown'] * batch_size
    else:
        batch_dataset_names = None
    
    return batch_dataset_ids, batch_params, batch_dataset_names

def to_python_float(value):
    """将tensor或数值转换为Python float"""
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    config_model=None,
    anchor_boxes=None,
    model_ema=None,
    ):
    
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("box_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("conf_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("category_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch_item in enumerate(              
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        if data_iter_step % accum_iter == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
        try:
            if isinstance(batch_item, dict):
                # 统一字典格式处理
                samples = batch_item.get('data', batch_item.get('samples', batch_item[0] if isinstance(batch_item, (list, tuple)) else batch_item))
                labels = batch_item.get('label', batch_item[1] if isinstance(batch_item, (list, tuple)) and len(batch_item) > 1 else None)
                raw_boxes = batch_item.get('raw_boxes', batch_item[2] if isinstance(batch_item, (list, tuple)) and len(batch_item) > 2 else None)
                batch_dataset_ids = batch_item.get('dataset_id')
                batch_params = batch_item.get('dataset_params')
                batch_dataset_names = batch_item.get('dataset_name')
            else:
                # 兼容原有格式 (samples, labels, raw_boxes)
                if isinstance(batch_item, (list, tuple)) and len(batch_item) >= 3:
                    samples, labels, raw_boxes = batch_item[0], batch_item[1], batch_item[2]
                else:
                    raise ValueError("Unsupported batch item format")
                batch_dataset_ids = None
                batch_params = None
                batch_dataset_names = None
                
            # 数据移到设备
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            raw_boxes = raw_boxes.to(device, non_blocking=True)
            
        except Exception as e:
            print(f"数据格式解析错误: {e}")
            continue

        with torch.amp.autocast("cuda"):
            # 处理批次参数 - 与预训练保持一致
            batch_dataset_ids, batch_params, batch_dataset_names = process_batch_params(
                samples.shape, batch_dataset_ids, batch_params, batch_dataset_names
            )
            # 前向传播 - 支持物理位置编码
            if batch_dataset_ids is not None and batch_params is not None:
                batch_dataset_ids = batch_dataset_ids.to(device, non_blocking=True)
                feature = model(samples, batch_dataset_ids=batch_dataset_ids, batch_params=batch_params)
            else:
                feature = model(samples)
                
            pred_raw, pred = decodeYolo_fpn(feature,
                                            input_size=input_size,
                                            anchor_boxes=anchor_boxes,
                                            scale=config_model["yolohead_xyz_scales"][0])

            box_loss, conf_loss, category_loss = criterion(pred_raw, pred, labels, raw_boxes[..., :6])
            loss = box_loss + conf_loss + category_loss

        loss_value = loss.item()    

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        total_loss_r, box_loss_r, conf_loss_r, category_loss_r = loss.cpu().detach(), \
            box_loss.cpu().detach(), conf_loss.cpu().detach(), category_loss.cpu().detach()
            
        # 分布式reduce
        total_loss_r = utils.all_reduce_mean(total_loss_r)
        box_loss_r = utils.all_reduce_mean(box_loss_r)
        conf_loss_r = utils.all_reduce_mean(conf_loss_r)
        category_loss_r = utils.all_reduce_mean(category_loss_r)
            
        # 更新指标
        metric_logger.update(lr=lr)
        metric_logger.update(loss=total_loss_r)
        metric_logger.update(box_loss=box_loss_r)
        metric_logger.update(conf_loss=conf_loss_r)
        metric_logger.update(category_loss=category_loss_r)

        loss_value_reduce = utils.all_reduce_mean(total_loss_r)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('reduce_train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            log_writer.add_scalar("loss/total_loss", total_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/box_loss", box_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/conf_loss", conf_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/category_loss", category_loss_r, global_step=epoch_1000x)

        if args.swanlab and utils.is_main_process():
            swanlab.log({
                'lr': to_python_float(lr),
                'total_loss': to_python_float(total_loss_r),
                'box_loss': to_python_float(box_loss_r),
                'conf_loss': to_python_float(conf_loss_r),
                'category_loss': to_python_float(category_loss_r)
            })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(    
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    log_writer=None,
    config_model=None,
    anchor_boxes=None,
    num_classes=None,
    args=None,
    model_ema=None
    ):

    eval_model = model
    if model_ema is not None:
        print("Using EMA model for evaluation.")
        eval_model = model_ema.ema  # 使用EMA的影子模型
    eval_model.eval()
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    
    metric_logger.add_meter("val_total_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("val_box_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("val_conf_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("val_category_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # 初始化评估指标 - 恢复原来的初始化方式
    ap_all_class = []
    for class_id in range(num_classes):
        ap_all_class.append([])
        
    mean_ap_test = 0.0
    total_losses = []
    box_losses = []
    conf_losses = []
    category_losses = []

    print_freq = 10
    
    for data_iter_step, batch_item in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):
        
        # 处理数据 - 与训练保持一致
        try:
            if isinstance(batch_item, dict):
                samples = batch_item.get('data', batch_item.get('samples', batch_item[0] if isinstance(batch_item, (list, tuple)) else batch_item))
                labels = batch_item.get('label', batch_item[1] if isinstance(batch_item, (list, tuple)) and len(batch_item) > 1 else None)
                raw_boxes = batch_item.get('raw_boxes', batch_item[2] if isinstance(batch_item, (list, tuple)) and len(batch_item) > 2 else None)
                batch_dataset_ids = batch_item.get('dataset_id')
                batch_params = batch_item.get('dataset_params')
                batch_dataset_names = batch_item.get('dataset_name')
            else:
                if isinstance(batch_item, (list, tuple)) and len(batch_item) >= 3:
                    samples, labels, raw_boxes = batch_item[0], batch_item[1], batch_item[2]
                else:
                    raise ValueError("Unsupported batch item format")
                batch_dataset_ids = None
                batch_params = None
                batch_dataset_names = None
                
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            raw_boxes = raw_boxes.to(device, non_blocking=True)
            
        except Exception as e:
            print(f"数据格式解析错误: {e}")
            continue


        if batch_dataset_ids is not None and batch_params is not None:
            batch_dataset_ids = batch_dataset_ids.to(device, non_blocking=True)
            if isinstance(batch_params, dict):
                batch_params = [batch_params] * samples.shape[0]
            feature = eval_model(samples, batch_dataset_ids=batch_dataset_ids, batch_params=batch_params)
        else:
            feature = eval_model(samples)
            
        pred_raw, pred = decodeYolo_fpn(feature,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
     
        box_loss, conf_loss, category_loss = criterion(pred_raw, pred, labels, raw_boxes[..., :6])
        loss = box_loss + conf_loss + category_loss
        
        # 记录损失
        total_losses.append(loss.item())
        box_losses.append(box_loss.item())
        conf_losses.append(conf_loss.item())
        category_losses.append(category_loss.item())
        
        # 恢复原来的mAP计算方式
        pred_np = pred.cpu().detach().numpy()
        raw_boxes_np = raw_boxes.cpu().numpy()
        
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes_np[batch_id]
            pred_frame = pred_np[batch_id]

            predictions = yoloheadToPredictions(pred_frame,
                                               conf_threshold=config_model["confidence_threshold"])

            nms_pred = nms(
                predictions, 
                config_model["nms_iou3d_threshold"],
                config_model["input_shape"],
                sigma=0.3,
                method="nms"
            )
            
            # 恢复原来的mAP计算调用
            mean_ap, ap_all_class = mAP.mAP(
                nms_pred,
                raw_boxes_frame,
                config_model["input_shape"],
                ap_all_class,
                tp_iou_threshold=config_model["mAP_iou3d_threshold"]
            )
            mean_ap_test += mean_ap

        metric_logger.update(val_total_loss=loss.item())
        metric_logger.update(val_box_loss=box_loss.item())
        metric_logger.update(val_conf_loss=conf_loss.item())
        metric_logger.update(val_category_loss=category_loss.item())
    
    # 恢复原来的分布式mAP处理
    mean_ap_test, ap_all_class_test = dist_decouple_mAP(ap_all_class)

    # 打印结果
    print("-------> ap_all: %.6f, ap_person: %.6f, ap_bicycle: %.6f, ap_car: %.6f, ap_motorcycle: %.6f, ap_bus: %.6f, "
          "ap_truck: %.6f" % (mean_ap_test, ap_all_class_test[0], ap_all_class_test[1], ap_all_class_test[2],
                              ap_all_class_test[3], ap_all_class_test[4], ap_all_class_test[5]))
    print("-------> total_loss: %.6f, box_loss: %.6f, conf_loss: %.6f, category_loss: %.6f" %
          (np.mean(total_losses), np.mean(box_losses), np.mean(conf_losses), np.mean(category_losses)))
    
    # 构建返回字典
    map_dict = {
        'ap_all': float(mean_ap_test),
        'ap_person': float(ap_all_class_test[0]),
        'ap_bicycle': float(ap_all_class_test[1]),
        'ap_car': float(ap_all_class_test[2]),
        'ap_motorcycle': float(ap_all_class_test[3]),
        'ap_bus': float(ap_all_class_test[4]),
        'ap_truck': float(ap_all_class_test[5]),
        'val_total_loss': float(np.mean(total_losses)),
        'val_box_loss': float(np.mean(box_losses)),
        'val_conf_loss': float(np.mean(conf_losses)),
        'val_category_loss': float(np.mean(category_losses))
    }
    
    # 日志记录
    if log_writer is not None:
        log_writer.add_scalar('ap/ap_all', map_dict['ap_all'], epoch)
        log_writer.add_scalar('ap/ap_person', map_dict['ap_person'], epoch)
        log_writer.add_scalar('ap/ap_bicycle', map_dict['ap_bicycle'], epoch)
        log_writer.add_scalar('ap/ap_car', map_dict['ap_car'], epoch)
        log_writer.add_scalar('ap/ap_motorcycle', map_dict['ap_motorcycle'], epoch)
        log_writer.add_scalar('ap/ap_bus', map_dict['ap_bus'], epoch)
        log_writer.add_scalar('ap/ap_truck', map_dict['ap_truck'], epoch)
        
        log_writer.add_scalar('val_loss/total_loss', map_dict['val_total_loss'], epoch)
        log_writer.add_scalar('val_loss/box_loss', map_dict['val_box_loss'], epoch)
        log_writer.add_scalar('val_loss/conf_loss', map_dict['val_conf_loss'], epoch)
        log_writer.add_scalar('val_loss/category_loss', map_dict['val_category_loss'], epoch)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if args.swanlab and utils.is_main_process():
        swanlab.log({
            'ap/all': map_dict['ap_all'],
            'ap/person': map_dict['ap_person'],
            'ap/bicycle': map_dict['ap_bicycle'],
            'ap/car': map_dict['ap_car'],
            'ap/motorcycle': map_dict['ap_motorcycle'],
            'ap/bus': map_dict['ap_bus'],
            'ap/truck': map_dict['ap_truck'],
            'val_loss/total': map_dict['val_total_loss'],
            'val_loss/box': map_dict['val_box_loss'],
            'val_loss/conf': map_dict['val_conf_loss'],
            'val_loss/category': map_dict['val_category_loss'],
        })
    
    # 统一返回格式
    stats_dict = {**map_dict, **{k: meter.global_avg for k, meter in metric_logger.meters.items()}}
    return stats_dict

def dist_decouple_mAP(ap_all_class__):
    """分布式mAP计算 - 保持原来的实现"""
    ap_all_class_test__ = []
    for ap_class_i in ap_all_class__:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = torch.tensor(np.sum(ap_class_i)).cuda()
            class_num = torch.tensor(len(ap_class_i)).cuda()
            dist.all_reduce(class_ap)
            dist.all_reduce(class_num)
            class_ap = class_ap / class_num
        ap_all_class_test__.append(class_ap.item())
    mean_ap_test__ = np.mean(ap_all_class_test__)
    
    return mean_ap_test__, ap_all_class_test__