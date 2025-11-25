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
# 在文件的顶部（import 之后）添加这个类
import time
import torch

# Place this class at the top of your engine_finetune_fpn.py file
import time
import torch


def train_one_epoch(
    model: torch.nn.Module,
    model_ema,
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
    ):
    
    model.train(True)
    
    # 将固定参数提前准备好
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

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
        
        samples = batch_item['data'].to(device, non_blocking=True)
        label0 = batch_item['label0'].to(device)
        label1 = batch_item['label1'].to(device)
        label2 = batch_item['label2'].to(device)
        raw_boxes = batch_item['raw_boxes'].to(device)
        condition = batch_item['condition'].to(device, non_blocking=True)
        raw_params = {k: v.to(device, non_blocking=True) for k, v in batch_item['raw_params'].items()}

        with torch.cuda.amp.autocast():
            feature0, feature1, feature2 = model(samples, condition=condition, batch_params=raw_params)
                
            pred_raw0, pred0 = decodeYolo_fpn(feature0,
                                            input_size=input_size,
                                            anchor_boxes=anchor_boxes,
                                            scale=config_model["yolohead_xyz_scales"][0])
            pred_raw1, pred1 = decodeYolo_fpn(feature1,
                                            input_size=input_size,
                                            anchor_boxes=anchor_boxes,
                                            scale=config_model["yolohead_xyz_scales"][0])
            pred_raw2, pred2 = decodeYolo_fpn(feature2,
                                            input_size=input_size,
                                            anchor_boxes=anchor_boxes,
                                            scale=config_model["yolohead_xyz_scales"][0])
            box_loss0, conf_loss0, category_loss0 = criterion(pred_raw0, pred0, label0, raw_boxes[..., :6])
            box_loss1, conf_loss1, category_loss1 = criterion(pred_raw1, pred1, label1, raw_boxes[..., :6])
            box_loss2, conf_loss2, category_loss2 = criterion(pred_raw2, pred2, label2, raw_boxes[..., :6])
            
            loss = box_loss0 + conf_loss0 + category_loss0 + \
                   box_loss1 + conf_loss1 + category_loss1 + \
                   box_loss2 + conf_loss2 + category_loss2
            conf_loss = conf_loss0 + conf_loss1 + conf_loss2
            box_loss = box_loss0 + box_loss1 + box_loss2
            category_loss = category_loss0 + category_loss1 + category_loss2
            
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
            model_ema.update() 
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        total_loss_r = utils.all_reduce_mean(loss.item() * accum_iter)
        box_loss_r = utils.all_reduce_mean(box_loss.item())
        conf_loss_r = utils.all_reduce_mean(conf_loss.item())
        category_loss_r = utils.all_reduce_mean(category_loss.item())
            
        metric_logger.update(lr=lr)
        metric_logger.update(loss=total_loss_r)
        metric_logger.update(box_loss=box_loss_r)
        metric_logger.update(conf_loss=conf_loss_r)
        metric_logger.update(category_loss=category_loss_r)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss/total_loss", total_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/box_loss", box_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/conf_loss", conf_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar("loss/category_loss", category_loss_r, global_step=epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if args.swanlab and utils.is_main_process():
            swanlab.log({
                        'lr': lr,
                        'total_loss': total_loss_r,
                        'box_loss': box_loss_r,
                        'conf_loss': conf_loss_r,
                        'category_loss': category_loss_r
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
):
    # --- 1. 初始化和准备 ---
    model.eval()

    # 将固定参数提前准备好
    anchor_boxes_tensor = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    all_predictions_local = []
    all_ground_truths_local = []
    
    # --- 2. 在每个GPU上独立进行预测和后处理 ---
    for batch_item in metric_logger.log_every(data_loader, 10, header):
        samples = batch_item['data'].to(device, non_blocking=True)
        label0 = batch_item['label0'].to(device)
        label1 = batch_item['label1'].to(device)
        label2 = batch_item['label2'].to(device)
        raw_boxes = batch_item['raw_boxes'].to(device)
        condition = batch_item['condition'].to(device, non_blocking=True)
        raw_params = {k: v.to(device, non_blocking=True) for k, v in batch_item['raw_params'].items()}

        feature0, feature1, feature2 = model(samples, condition=condition, batch_params=raw_params)

        pred_raw0, pred0 = decodeYolo_fpn(feature0, input_size, anchor_boxes_tensor, config_model["yolohead_xyz_scales"][0])
        pred_raw1, pred1 = decodeYolo_fpn(feature1, input_size, anchor_boxes_tensor, config_model["yolohead_xyz_scales"][1])
        pred_raw2, pred2 = decodeYolo_fpn(feature2, input_size, anchor_boxes_tensor, config_model["yolohead_xyz_scales"][2])
        
        box_loss0, conf_loss0, category_loss0 = criterion(pred_raw0, pred0, label0, raw_boxes.to(device)[..., :6])
        box_loss1, conf_loss1, category_loss1 = criterion(pred_raw1, pred1, label1, raw_boxes.to(device)[..., :6])
        box_loss2, conf_loss2, category_loss2 = criterion(pred_raw2, pred2, label2, raw_boxes.to(device)[..., :6])

        loss = box_loss0 + conf_loss0 + category_loss0 + \
               box_loss1 + conf_loss1 + category_loss1 + \
               box_loss2 + conf_loss2 + category_loss2
        
        metric_logger.update(val_total_loss=loss.item())
        
        # 解码和后处理 (NMS)，将结果移到CPU
        pred0 = pred0.cpu().numpy()
        pred1 = pred1.cpu().numpy()
        pred2 = pred2.cpu().numpy()
        raw_boxes_np = raw_boxes.cpu().numpy()

        for i in range(len(raw_boxes_np)):
            preds = [
                yoloheadToPredictions(pred0[i], config_model["confidence_threshold"]),
                yoloheadToPredictions(pred1[i], config_model["confidence_threshold"]),
                yoloheadToPredictions(pred2[i], config_model["confidence_threshold"]),
            ]
            predictions_cat = np.concatenate(preds, axis=0)

            nms_pred = nms(
                predictions_cat, 
                config_model["nms_iou3d_threshold"],
                config_model["input_shape"],
                sigma=0.3,
                method="nms"
            )
            
            all_predictions_local.append(nms_pred)
            all_ground_truths_local.append(raw_boxes_np[i])

    # 同步所有进程的损失统计
    metric_logger.synchronize_between_processes()
    print("Averaged validation loss stats:", metric_logger)

    # --- 3. 分布式数据收集 ---
    # 使用 all_gather_object 收集所有进程的 Python 对象列表
    # This can be slow if the objects are very large, but it's the most straightforward way.
    all_predictions_dist = [None] * utils.get_world_size()
    all_ground_truths_dist = [None] * utils.get_world_size()

    if args.distributed:
        dist.barrier() # Ensure all processes have finished prediction
        dist.all_gather_object(all_predictions_dist, all_predictions_local)
        dist.all_gather_object(all_ground_truths_dist, all_ground_truths_local)
    else:
        all_predictions_dist = [all_predictions_local]
        all_ground_truths_dist = [all_ground_truths_local]

    map_dict = {}
    if utils.is_main_process():
        print("Rank 0: All predictions gathered. Starting final mAP calculation...")
        final_preds = [item for sublist in all_predictions_dist for item in sublist]
        final_gts = [item for sublist in all_ground_truths_dist for item in sublist]

        ap_all_class_list = [[] for _ in range(num_classes)]
        
        for i in range(len(final_preds)):
            mAP.mAP(
                final_preds[i],
                final_gts[i],
                config_model["input_shape"],
                ap_all_class_list,
                tp_iou_threshold=config_model["mAP_iou3d_threshold"]
            )
        
        ap_per_class = []
        for class_aps in ap_all_class_list:
            if len(class_aps) == 0:
                ap_per_class.append(0.0)
            else:
                ap_per_class.append(np.mean(class_aps))
        
        mean_ap = np.mean(ap_per_class)

        print("-------> ap_all: %.6f, ap_person: %.6f, ap_bicycle: %.6f, ap_car: %.6f, ap_motorcycle: %.6f, ap_bus: %.6f, "
              "ap_truck: %.6f" % (mean_ap, ap_per_class[0], ap_per_class[1], ap_per_class[2],
                                  ap_per_class[3], ap_per_class[4], ap_per_class[5]))
        
        map_dict = {
            'ap_all': mean_ap,
            'ap_person': ap_per_class[0], 'ap_bicycle': ap_per_class[1],
            'ap_car': ap_per_class[2], 'ap_motorcycle': ap_per_class[3],
            'ap_bus': ap_per_class[4], 'ap_truck': ap_per_class[5],
        }

    # --- 5. 将结果广播到所有进程 ---
    if args.distributed:
        results_to_broadcast = torch.zeros(num_classes + 1, device=device) # +1 for mean_ap
        if utils.is_main_process():
            results_to_broadcast[0] = map_dict['ap_all']
            results_to_broadcast[1:] = torch.tensor(
                [map_dict['ap_person'], map_dict['ap_bicycle'], map_dict['ap_car'],
                 map_dict['ap_motorcycle'], map_dict['ap_bus'], map_dict['ap_truck']],
                device=device
            )
        
        dist.broadcast(results_to_broadcast, src=0)
        torch.cuda.synchronize() # Wait for broadcast to finish

        # 所有进程从广播的Tensor中更新map_dict
        results_list = results_to_broadcast.cpu().tolist()
        map_dict = {
            'ap_all': results_list[0], 'ap_person': results_list[1],
            'ap_bicycle': results_list[2], 'ap_car': results_list[3],
            'ap_motorcycle': results_list[4], 'ap_bus': results_list[5],
            'ap_truck': results_list[6],
        }

    # --- 6. 整合并返回最终结果 ---
    # 补充损失信息
    map_dict.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})

    # 日志记录 (主进程执行)
    if utils.is_main_process():
        if log_writer is not None:
            for k, v in map_dict.items():
                log_writer.add_scalar(f'ap/{k}' if 'ap' in k else f'val_loss/{k}', v, epoch)
        
        if args.swanlab:
            swanlab_log = {f'val/{k}': v for k, v in map_dict.items()}
            swanlab.log(swanlab_log, step=epoch)



    return map_dict
    
def dist_decouple_mAP(ap_all_class__):
    ap_all_class_test__ = []
    for ap_class_i in ap_all_class__:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap_sum = torch.tensor(np.sum(ap_class_i), device='cuda')
            class_num = torch.tensor(len(ap_class_i), device='cuda')
            if dist.is_initialized():
                dist.all_reduce(class_ap_sum)
                dist.all_reduce(class_num)
            ap = class_ap_sum / class_num if class_num > 0 else 0.0
            class_ap = ap.item()
        ap_all_class_test__.append(class_ap)
    mean_ap_test__ = np.mean(ap_all_class_test__)
    
    return mean_ap_test__, ap_all_class_test__