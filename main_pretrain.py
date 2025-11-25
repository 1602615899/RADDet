import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import sys


sys.path.append('./datasets/')
sys.path.append('./models/')

from datasets.CRUW import Create_CRUW_Pretrain_Dataset
import utils 
from utils import NativeScalerWithGradNormCount as NativeScaler
from datasets import RADDet
from torch.utils.data import ConcatDataset
from datasets.RADDet import Create_RADDet_Pretrain_Dataset
from datasets.CARRADA import Create_CARRADA_Pretrain_Dataset
from datasets.K_Radar import Create_K_Radar_Pretrain_Dataset
from datasets.RaDICaL import Create_RaDICaL_indoor_Dataset, Create_RaDICaL_outdoor30_Dataset, Create_RaDICaL_outdoor60_Dataset, Create_RaDICaL_highRes_Dataset
from datasets.SCORP import Create_SCORP_Pretrain_Dataset 

import models.pretrain.models_pretrain as models_arm
from engine_pretrain import train_one_epoch
import swanlab  
import torch.distributed as dist
import shutil
import logging


def get_args_parser():
    parser = argparse.ArgumentParser('AR pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints) \
                        ')

    # model parameters
    parser.add_argument('--model', default='vision_mamba_3d_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--autoreg_dim', default='doppler', type=str,
                        choices=['doppler', 'range', 'angle'],
                        help="Dimension for auto-regressive pre-training ('doppler', 'range', 'angle')")
    
    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # dataset parameters
    
    parser.add_argument('--RADDet_config_path', default='./datasets/RADDet_config.json', type=str,
                        help='RADDet config path')
    
    parser.add_argument('--CRUW_config_path', default='/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/config_Rodnet.py', type=str,
                        help='CRUW config path')
    
    parser.add_argument('--SCORP_data_path', default='/media/ljm/Raid/ChenHongliang/20190813_scorp_dataset', type=str, help='SCORP dataset root path')
    parser.add_argument('--K_Radar_data_path', default='/media/ljm/Raid/ChenHongliang/K-Radar_reprocessed_HL_DPFT', type=str, help='K-Radar dataset root path')

    parser.add_argument('--sensor_config_path', default='/media/ljm/Raid/ChenHongliang/RAGM/datasets/sensor_params.json', type=str,
                        help='传感器参数配置文件路径')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to save logs, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='prefetch_factor for data loader')
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')



    parser.add_argument('--checkpoint_period', default=1, type=int, help='checkpoint_period')


    parser.add_argument('--bias_wd', action='store_true', default=False, help='weight decay on bias')

    parser.add_argument('--comment', default=None, type=str, help='comments for logging')

    parser.add_argument('--data_type', default='ALL', type=str, help='data type to train on')

    parser.add_argument("--swanlab", action="store_true", default=False, help="是否使用swanlab进行记录")


    return parser


def main(args):

    utils.init_distributed_mode(args)

    # print args
    print(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    if args.swanlab and utils.is_main_process():
        swanlab.init(
            description=args.comment,
            config=args,
            logdir="./swanlab_dir",
            project="xxx",
        )
                

    # set device
    if args.distributed:
        device = torch.device(args.gpu)
    else:
        device = torch.device(args.device)

    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.swanlab and utils.is_main_process():
        code_files = [
            "main_pretrain.py", 
            "engine_pretrain.py",        
            "models/pretrain/models_pretrain.py", 
            "models/pretrain/mamba_simple.py",
            "models/pretrain/selective_scan_interface.py",
            "datasets/RADDet.py",
            "datasets/CARRADA.py",
            "datasets/CRUW.py",
            "datasets/SCORP.py",
            "run_pretrain.sh",
        ]
        
        # 创建代码快照目录
        code_snapshot_dir = os.path.join(args.output_dir, "code_snapshots")
        os.makedirs(code_snapshot_dir, exist_ok=True)
        
        # 备份并记录代码文件
        for src_path in code_files:
            if os.path.exists(src_path):
                shutil.copy2(src_path, code_snapshot_dir)
                
        print(f"Model code snapshot saved to: {code_snapshot_dir}")

    # create datasets
    if args.data_type == 'RADDet':
        dataset_train = Create_RADDet_Pretrain_Dataset(args.RADDet_config_path, args.sensor_config_path)
    elif args.data_type == 'CARRADA':
        dataset_train = Create_CARRADA_Pretrain_Dataset(args.sensor_config_path)
    elif args.data_type == 'CRUW':
        dataset_train = Create_CRUW_Pretrain_Dataset(args.CRUW_config_path, args.sensor_config_path)
    elif args.data_type == 'SCORP':
        dataset_train = Create_SCORP_Pretrain_Dataset(sensor_config_path=args.sensor_config_path)
    elif args.data_type == 'K-Radar':
        dataset_train = Create_K_Radar_Pretrain_Dataset(args.K_Radar_data_path, args.sensor_config_path)
    elif args.data_type == 'RaDICaL-indoor':
        dataset_train = Create_RaDICaL_indoor_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
    elif args.data_type == 'RaDICaL-outdoor30':
        dataset_train = Create_RaDICaL_outdoor30_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
    elif args.data_type == 'RaDICaL-outdoor60':
        dataset_train = Create_RaDICaL_outdoor60_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
    elif args.data_type == 'RaDICaL-highRes':
        dataset_train = Create_RaDICaL_highRes_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
    elif args.data_type == 'ALL':
        raddet_dataset = Create_RADDet_Pretrain_Dataset(args.RADDet_config_path, args.sensor_config_path)
        # print("raddet num:",len(raddet_dataset))
        carrada_dataset = Create_CARRADA_Pretrain_Dataset(args.sensor_config_path)
        # print("carrada num:",len(carrada_dataset))
        scorp_dataset = Create_SCORP_Pretrain_Dataset(args.SCORP_data_path, sensor_config_path=args.sensor_config_path)
        # print("scorp num:",len(scorp_dataset))
        cruw_dataset = Create_CRUW_Pretrain_Dataset(args.CRUW_config_path, args.sensor_config_path)
        # print("cruw num:",len(cruw_dataset))
        # k_radar_dataset = Create_K_Radar_Pretrain_Dataset(args.K_Radar_data_path, args.sensor_config_path)
        # print("k_radar num:",len(k_radar_dataset))
        # radical_indoor_dataset = Create_RaDICaL_indoor_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
        # print("radical_indoor num:",len(radical_indoor_dataset))
        # radical_outdoor30_dataset = Create_RaDICaL_outdoor30_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
        # print("radical_outdoor30 num:",len(radical_outdoor30_dataset))
        # radical_outdoor60_dataset = Create_RaDICaL_outdoor60_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
        # print("radical_outdoor60 num:",len(radical_outdoor60_dataset))
        # radical_highres_dataset = Create_RaDICaL_highRes_Dataset(args.RaDICaL_data_path, args.sensor_config_path)
        # print("radical_highres num:",len(radical_highres_dataset))
        
        dataset_all = []
        dataset_all.append(raddet_dataset)
        dataset_all.append(carrada_dataset)
        dataset_all.append(scorp_dataset)
        dataset_all.append(cruw_dataset)
        # dataset_all.append(k_radar_dataset)
        # dataset_all.append(radical_indoor_dataset)
        # dataset_all.append(radical_outdoor30_dataset)
        # dataset_all.append(radical_outdoor60_dataset)
        # dataset_all.append(radical_highres_dataset)
        dataset_train = ConcatDataset(dataset_all)
        print("dataset_all num:",len(dataset_all))
    else:
        raise ValueError("Invalid data_type: %s" % args.data_type)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        prefetch_factor=4
    )

    print(f"Creating model: {args.model} with autoreg_dim='{args.autoreg_dim}'")
    model = models_arm.__dict__[args.model](autoreg_dim=args.autoreg_dim)
    print(model)
    model.to(device)

    model_without_ddp = model

    # effective batch size = batch_size * accum_iter（梯度累积的迭代次数） * ddp进程总数
    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    # absolute_lr = base_lr * total_batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params: {} M'.format(n_parameters / 1e6))

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu], 
                                                          find_unused_parameters=False
                                                          )
        model_without_ddp = model.module

    param_groups = utils.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=args.bias_wd)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    utils.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    best_loss = 100
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
            filename = f'checkpoint-{epoch}.pth'
            
            # Call the save_model utility with the specific filename
            utils.save_model(args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch,
                            filename=filename)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 'n_parameters': n_parameters}
        
        if args.swanlab and utils.is_main_process():
                swanlab.log({
                    'epoch': epoch,
                    'train_loss': log_stats['train_loss'],
                    'lr': optimizer.param_groups[0]['lr'],
                    'best_loss': best_loss,
                    'n_parameters': n_parameters
                })

        if log_stats['train_loss'] < best_loss:
            best_loss = log_stats['train_loss']
            best_epoch = epoch
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                },
                best_model_path
            )
            print(f"New best model saved with loss: {best_loss:.4f}")

        log_stats.update({'best_loss': best_loss, 'best_epoch': best_epoch})

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.swanlab:
                swanlab.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print(torch.cuda.memory_allocated())
        
    if args.swanlab:
        swanlab.finish()



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # 给文件夹加上时间戳
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.output_dir.endswith('/'):
            args.output_dir += '/'
    args.output_dir += current_time + '_' + args.model + '_pretrain' + '_' + args.data_type
    args.log_dir = args.output_dir

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    # 保存配置文件
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            f.write(json.dumps(vars(args)))

    # 配置日志
    log_file = os.path.join(args.output_dir, "output.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    sys.stdout = utils.StreamToLogger(logging.getLogger(), logging.INFO)
    logging.info(f"Starting the script with args: {args}")


    main(args)

