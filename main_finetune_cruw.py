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

from models.CRUW_finetune.rodnet.datasets.collate_functions import cr_collate 
sys.path.append('./datasets/')
sys.path.append('./models/')

from models.CRUW_finetune.CRUW_finetune_dataset import CRUW, CRDataset, Create_CRUW_Finetune_Dataset

import utils 
from utils import NativeScalerWithGradNormCount as NativeScaler

from models.CRUW_finetune.load_configs import load_configs_from_file, parse_cfgs, update_config_dict

from mvrss.learners.initializer import Initializer

import models.finetune_cruw as models_cruw
from engine_finetune_cruw import train_one_epoch, evaluate
from torch.nn.init import kaiming_normal_
import torch.nn as nn
import swanlab

import logging


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fituning', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints) \
                        梯度累积的次数，在显存受限的情况下，通过累积多个小批次的梯度，可以达到更大的等效批次大小，从而模拟大批次训练。')

    # Model parameters
    parser.add_argument('--model', default='RADTR_cruw_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
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

    # Dataset parameters
    parser.add_argument('--CRUW_config_path', default='./models/CRUW_finetune/config_Rodnet.py', type=str,
                        help='CRUW_config_path')
    parser.add_argument('--output_dir', default='./ft_CRUW_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',     # 主gpu
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--prefetch_factor', default=2, type=int)

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--distributed", action="store_true", default=False, 
                        help="是否启用分布式训练，启用则多GPU或多节点，否则单GPU")
    parser.add_argument('--world_size', default=1, type=int,    # 这里设置好像没用
                        help='number of distributed processes, 整个训练系统中的进程总数，所有节点上的GPU数量')
    parser.add_argument('--local_rank', default=-1, type=int,   # 这里设置好像没用，因为DDP会自己设置local_rank
                        help='node rank for distributed training, 在单个节点（机器）中的进程编号，从0开始')
    parser.add_argument('--dist_on_itp', action='store_true', help='是否在ITP上运行分布式训练')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training，用于初始化分布式训练的URL，确保DDP的不同进程之间能够相互通信和同步。')

    # mae_rpt parameters
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--bias_wd", action="store_true", help="是否对bias参数进行weight decay")

    parser.add_argument("--comment", type=str, help="对本次实验的改动描述，或想探究的点")

    parser.add_argument("--data_type", type=str, default="CRUW", help="数据集", choices=["RADDet", "CARRADA", "CRUW","ALL"])

    parser.add_argument("--swanlab", action="store_true", default=False, help="是否使用swanlab进行记录")
    
    # evaluation parameters
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--epoch', default=29, type=int)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

     
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )

    parser.add_argument('--use_film_metadata', action='store_true', default=False,
                        help='Whether to use FiLM metadata for conditioning')
    parser.add_argument('--bimamba_type', type=str, default='v4', choices=['v1', 'v2', 'v3', 'v4', 'none'],
                        help='Type of BiMamba to use if any')

    return parser


def main(args):

    utils.init_distributed_mode(args)

    # print args
    # print(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.swanlab:
        # 替换wandb.init为swanlab.init
        swanlab.init(
            description=args.comment,
            config=args,
            logdir="./swanlab_dir",
            project="mae_finetune_RADDet",
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


    # load dataset and config
    config_dict = load_configs_from_file(args.CRUW_config_path)

    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], 
                   sensor_config_name='/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/cruw/dataset_configs/sensor_config_rod2021.json',
                   object_config_name='/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/cruw/dataset_configs/object_config.json')
    dataset_train = Create_CRUW_Finetune_Dataset(dataset, config_dict)
    print(len(dataset_train))


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        # if args.dist_eval:
        #     if len(dataset_val) % num_tasks != 0:
        #         print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #               'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #               'equal num of samples per-process.')
        #     sampler_val = torch.utils.data.DistributedSampler(
        #         dataset_val, num_replicas=num_tasks, rank=global_rank, drop_last=False, shuffle=False)  # shuffle=True to reduce monitor bias
        # else:
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val) # Samples elements sequentially, always in the same order.
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
        except Exception as e:
            print(e)
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
        persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存
        # prefetch_factor=args.prefetch_factor,
        # prefetch_factor=2
        # collate_fn=cr_collate
    )

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val,
    #     sampler=sampler_val,
    #     batch_size=1,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False,
    #     persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存
    #     # collate_fn=cr_collate
    # )


    model = models_cruw.__dict__[args.model](
        use_film_metadata=args.use_film_metadata,
        bimamba_type=args.bimamba_type,
    )   
    
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        
        head_keys = [
            "yolo_head.conv1.weight", "yolo_head.conv1.bias",
            "yolo_head.bn1.weight", "yolo_head.bn1.bias",       # 这里用batch normalization可以吗？前面用的都是layer normalization
            "yolo_head.conv2.weight", "yolo_head.conv2.bias",
            # "patch_embed.proj.weight", "patch_embed.proj.bias",     # for  2 channels input
        ]
        # checkpoint_model 保存预训练模型的模型参数及权重，state_dict对应微调模型的模型参数
        for k in head_keys:      # 这里的head是什么？ 应该把decoder 的参数删除吧
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        # 把模型分成两块，这里的模型权重好像就不能匹配了，好像还是要写在一起
                
        # interpolate position embedding
        utils.interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model    strict=False 允许权重部分匹配，decoder部分不会被加载
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        
        # didn't manually initialize yolo head, need?
        
    if args.finetune and args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
                
        # 把模型分成两块，这里的模型权重好像就不能匹配了，好像还是要写在一起
                
        # interpolate position embedding
        utils.interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model    strict=False 允许权重部分匹配，decoder部分不会被加载
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # effective batch size = batch_size * accum_iter（梯度累积的迭代次数） * ddp进程总数
    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    # absolute_lr = base_lr * total_batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu],    # device_ids=[torch.cuda.current_device()]
                                                          find_unused_parameters=False       # change True to False
                                                          )
        model_without_ddp = model.module

    # # 这里用的还是pretrain的params，mae的这里加了no weight decay list
    param_groups = utils.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=args.bias_wd,
                                        #   skip_list=['pos_embed']
                                          )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)     # pretrain里设置了betas=(0.9, 0.95)，这里用默认
    print(optimizer)
    
    # # follow RADDet 的原始训练来做,Adam optimizer
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    
    loss_scaler = NativeScaler()
    # loss_scaler = torch.GradScaler()

    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
        
    utils.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    
    if args.eval:
        test_stats = evaluate(
            model=model,
            criterion=criterion,
            device=device,
            epoch=args.epoch,
            dataset = dataset,
            log_writer=log_writer,
            config_model=config_dict,
            args = args,
            )
        
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            config_model=config_dict,
            dataset = dataset,
            args=args,
        )
            
        # validation
        # print(f"epochs: {epoch}, start validation")
        # test_stats = evaluate(
        #     model=model,
        #     criterion=criterion,
        #     device=device,
        #     epoch=epoch,
        #     dataset = dataset,
        #     log_writer=log_writer,
        #     config_model=config_dict,
        #     args = args,
            
        #     )
             
        # save best
        # if test_stats['range_doppler']['miou'] > best_test_miou:
        #     best_test_miou = test_stats['range_doppler']['miou']
        #     best_model_path = os.path.join(args.output_dir, "best_model.pth")
        #     torch.save(
        #         {
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'epoch': epoch,
        #             'scaler': loss_scaler.state_dict(),
        #             'args': args,
        #         },
        #         best_model_path
        #     )
        #     print(f"New best model saved with miou: {best_test_miou:.4f}")
            
            
        # save model periodically
        if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
            utils.save_model(args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch)
            
        # if log_writer is not None:
        #     log_writer.add_scalar('test_ap_all', test_stats['ap_all'], epoch)
            
        # log_stats = {
        #     "epoch": epoch,
        #     # 训练指标（假设train_stats是平铺字典）
        #     **{f"train_{k}": round(v, 6) for k, v in train_stats.items() 
        #     if isinstance(v, (int, float))},
        #     # 测试指标（明确指定嵌套字典中的值）
        #     "test_rd_loss": round(test_stats['range_doppler'].get('loss', 0), 6),
        #     "test_ra_loss": round(test_stats['range_angle'].get('loss', 0), 6),
        #     "test_rd_prec": round(test_stats['range_doppler'].get('prec', 0), 6),
        #     "test_ra_prec": round(test_stats['range_angle'].get('prec', 0), 6),
        #     "test_rd_miou": round(test_stats['range_doppler'].get('miou', 0), 6),
        #     "test_ra_miou": round(test_stats['range_angle'].get('miou', 0), 6),
        #     "n_parameters": n_parameters,
        #     "best_test_miou": best_test_miou
        # }


        # if args.output_dir and utils.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        #     if args.wandb:
        #         swanlab.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # print(torch.cuda.memory_allocated())



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # 给文件夹加上时间戳
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.output_dir.endswith('/'):
            args.output_dir += '/'
    args.output_dir += current_time + '_' + args.model + '_finetune'
    if args.eval:
        args.output_dir += '_eval'
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