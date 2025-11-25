import math
import os
import sys
from typing import Iterable
import swanlab
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.CARRADA_finetune.utils.functions import get_metrics
from models.CARRADA_finetune.utils.metrics import Evaluator
from models.CRUW_finetune.CRUW_finetune_dataset import CRDataset
from models.CRUW_finetune.rodnet.core.post_processing.output_results import write_dets_results_single_frame
from models.CRUW_finetune.rodnet.core.post_processing.postprocess import post_process_single_frame
from models.CRUW_finetune.rodnet.core.radar_processing.chirp_ops import chirp_amp
# from models.CRUW_finetune.rodnet.utils.visualization.demo import visualize_test_img, visualize_test_img_wo_gt

from models.CRUW_finetune.rodnet.utils.visualization.demo import visualize_test_img, visualize_test_img_wo_gt, visualize_train_img
import utils
import torch.distributed as dist
import time
# 引入 matplotlib.pyplot 库用于保存图片和绘制
import matplotlib.pyplot as plt
import matplotlib # 引入 matplotlib 用于获取 Colormap 等
import matplotlib.colors as mcolors # 用于创建自定义 Colormap
from torch.amp import autocast
from models.CRUW_finetune.cruw import CRUW
from models.CRUW_finetune.cruw.eval import evaluate_rod2021

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    config_model=None,
    dataset=None,
    args=None,
    ):

    if len(data_loader) == 0:
        raise ValueError("DataLoader is empty! Check dataset paths and batch size.")
    
    print('Criterion:', criterion)

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # add metrics
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("total_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)
    print_freq = 20     # 每20个batch打印一次
    radar_configs = dataset.sensor_cfg.radar_cfg
    accum_iter = args.accum_iter if hasattr(args, 'accum_iter') else 1

    out_path = os.path.join(args.output_dir, 'train', f"{epoch:02d}")
    train_viz_path = os.path.join(out_path, 'train_viz')
    if utils.is_main_process():
        os.makedirs(train_viz_path, exist_ok=True)

    running_stats = {
        'total_loss': [],
    }

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, labels) in enumerate(             
    #     metric_logger.log_every(data_loader, print_freq, header)
    # ):
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if data_iter_step % accum_iter == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # return {
        #     'data': torch.tensor(RAD_data, dtype=torch.float32),
        #     'label': torch.tensor(confmap_gt, dtype=torch.float32),
        #     'seq_names':data_dict['seq_names'],
        #     'radar_npy_win': radar_npy_win,
        #     'image_paths': data_dict['image_paths'],
        #     'condition': self.condition_tensor, # The normalized tensor for the model
        #     'raw_params': self.raw_params       # The original dict for visualization & pos encoding
        # }
        samples = data['data']
        confmap_gt = data['label']
        seq_names = data['seq_names']
        radar_npy_win = data['radar_npy_win']
        condition = data['condition']
        raw_params = data['raw_params']
        image_paths = data['image_paths']

        optimizer.zero_grad()        
        # with autocast(device_type='cuda', dtype=torch.float16):
        confmap_preds = model(samples.float().cuda(), condition, raw_params)

        total_loss = criterion(confmap_preds, confmap_gt.float().cuda())

        running_stats['total_loss'].append(total_loss.item())
        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)
        total_loss /= accum_iter

        # loss_scaler.scale(total_loss).backward()
        # if (data_iter_step + 1) % accum_iter == 0:
        #     loss_scaler.step(optimizer)
        #     loss_scaler.update()
        #     optimizer.zero_grad()
        # total_loss.backward()  # 计算梯度
        
        loss_scaler(
            total_loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            # clip_grad=args.clip_grad,   # 梯度裁剪
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()    # 同步cuda 等待所有在 GPU 上进行的操作完成
        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"],
            total_loss=total_loss.item(),
        )

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)
            log_writer.add_scalar('loss/total', total_loss.item(), epoch_1000x)


        if args.swanlab and utils.is_main_process():
            swanlab.log({
                        'lr': lr,
                        'total_loss': total_loss_r,
                    })
            
        if data_iter_step % config_model['train_cfg']['log_step'] == 0:
                confmap_pred = confmap_preds.cpu().detach().numpy()
                chirp_amp_curr = chirp_amp(radar_npy_win.numpy()[0, :, 0, 0, :, :], radar_configs['data_type'])

                # draw train images
                fig_name = os.path.join(train_viz_path,
                                        '%03d_%06d.png' % (epoch + 1, data_iter_step + 1))
                img_path = image_paths[0][0]
                visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                    confmap_pred[0, :3, 0, :, :],
                                    confmap_gt[0, :3, 0, :, :])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("训练统计:")
    print(f"平均总损失: {np.mean(running_stats['total_loss']):.4f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from models.CRUW_finetune.rodnet.core.post_processing import ConfmapStack
@torch.no_grad()
def evaluate(    
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    dataset,
    log_writer=None,
    config_model=None,
    args = None,
    ):
    radar_configs = dataset.sensor_cfg.radar_cfg
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    running_losses = list()  
    print_freq = 10
    out_path = os.path.join(args.output_dir, f"{epoch:02d}")
    result_path = os.path.join(out_path, "evaluate")
    seq_names = ['2019_04_09_BMS1001', '2019_04_30_MLMS001', '2019_05_23_PM1S013', '2019_09_29_ONRD005']
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    for seq_name in seq_names:
        seq_res_dir = os.path.join(out_path, seq_name)
        if not os.path.exists(seq_res_dir):
            os.makedirs(seq_res_dir, exist_ok=True)
        seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
        if not os.path.exists(seq_res_viz_dir):
            os.makedirs(seq_res_viz_dir, exist_ok=True)
        f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
        f.close()

    for subset in seq_names:
        crdata_test = CRDataset(data_dir='/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/data', dataset=dataset, config_dict=config_model, split='test',
                                    noise_channel=False, subset=subset, is_random_chirp=False)
        sampler_val = torch.utils.data.SequentialSampler(crdata_test)

        data_loader_val = torch.utils.data.DataLoader(
            crdata_test,
            sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存
            # collate_fn=cr_collate
        )

        total_count = 0
        print(subset)
        confmap_shape = (3, 128, 128)
        init_genConfmap = ConfmapStack(confmap_shape)
        iter_ = init_genConfmap
    
        for i in range(config_model['train_cfg']['win_size'] - 1):
            while iter_.next is not None:
                iter_ = iter_.next
            iter_.next = ConfmapStack(confmap_shape)

        with torch.no_grad():
            for data_iter_step, data in enumerate(
                metric_logger.log_every(data_loader_val, print_freq, header)):
                
                samples = data[0]
                confmap_gt = data[1]

                # try:
                #     # 尝试获取RGB图像数据的文件路径，如果失败则打印警告信息
                #     image_paths = data['image_paths'][0]
                # except:
                #     print('warning: fail to load RGB images, will not visualize results')
                #     image_paths = None
                # samples = samples.permute(0, 3, 1, 2)  # NRAD -> NDRA   only for RADDet
                samples = samples.permute(0, 3, 1, 2)  # NRAD -> NDRA
                samples = samples.unsqueeze(1)

                seq_name = data[3][0]
                save_path = os.path.join(result_path, seq_name + '.txt')
                start_frame_id = data[2].item()
                # end_frame_id = data['end_frame'].item()
                confmap_pred = model(samples.float().cuda())

                confmap_pred = confmap_pred.cpu().detach().numpy()
                iter_ = init_genConfmap

                for i in range(confmap_pred.shape[2]):
                    if iter_.next is None and i != confmap_pred.shape[2] - 1:
                        iter_.next = ConfmapStack(confmap_shape)
                    
                    iter_.append(confmap_pred[0, :, i, :, :])
                    iter_ = iter_.next

                for i in range(config_model['test_cfg']['test_stride']):
                    total_count += 1
                    res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_model)
                    cur_frame_id = start_frame_id + i
                    write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                    # confmap_pred_0 = init_genConfmap.confmap
                    # res_final_0 = res_final
                    
                    # if data[5] is not None:
                    #     img_path = data[5][i][0]
                    #     radar_input = chirp_amp(data[4].numpy()[0, :, i, 0, :, :], radar_configs['data_type'])
                    #     fig_name = os.path.join(out_path, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                    #     if confmap_gt is not None:
                    #         confmap_gt_0 = confmap_gt[0, :, i, :, :]
                    #         visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0, dataset, sybl=False)
                    #     else:
                    #         visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0, dataset, sybl=False)
                    
                    init_genConfmap = init_genConfmap.next
                
                if iter == len(data_loader) - 1:
                    # 初始化偏移量和当前帧编号
                    offset = config_model['test_cfg']['test_stride']
                    cur_frame_id = start_frame_id + offset
                    
                    while init_genConfmap is not None:
                        total_count += 1
                        
                        res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_model)
                        
                        write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                        
                        # confmap_pred_0 = init_genConfmap.confmap
                        # res_final_0 = res_final
                        
                        # if data[5] is not None:
                        #     img_path = data[5][offset][0]
                        #     radar_input = chirp_amp(data[4].numpy()[0, :, offset, 0, :, :], radar_configs['data_type'])
                        #     fig_name = os.path.join(out_path, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                        #     if confmap_gt is not None:
                        #         confmap_gt_0 = confmap_gt[0, :, offset, :, :]
                        #         visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0, dataset, sybl=False)
                        #     else:
                        #         visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0, dataset, sybl=False)
                        
                        init_genConfmap = init_genConfmap.next
                        offset += 1
                        cur_frame_id += 1

                if init_genConfmap is None:
                    init_genConfmap = ConfmapStack(confmap_shape)
    
    metric_logger.update()
    submit_dir = result_path
    truth_dir = r"models/CRUW_finetune/gt"
    output_txt = os.path.join(out_path, "evaluation_results.txt")
    evaluate_rod2021(submit_dir, truth_dir, dataset, output_txt)


    return None


    