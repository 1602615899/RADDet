

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

import utils 
from utils import NativeScalerWithGradNormCount as NativeScaler
from models.RADDet_finetune.RADDet_finetune_dataset_fpn import Create_RADDet_Finetune_Dataset
from models.RADDet_finetune.yolo_loss import RadDetLoss

import models.RADDet_finetune.loader as loader

import models.finetune_raddet as models_raddet
from engine_finetune_fpn import train_one_epoch, evaluate
from torch.nn.init import kaiming_normal_
from torch_ema import ExponentialMovingAverage

import swanlab
import shutil
import logging


def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning RADDet', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='raddet_tiny', type=str, metavar='MODEL',
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

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--RADDet_config_path', default='./models/RADDet_finetune/config.json', type=str,
                        help='RADDet_config_path')
    parser.add_argument('--sensor_config_path', default='/media/ljm/Raid/ChenHongliang/RAGM/datasets/sensor_params.json', type=str,
                        help='传感器参数配置文件路径')
    parser.add_argument('--output_dir', default='./ft_output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./ft_output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
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
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_on_itp', action='store_true', help='是否在ITP上运行分布式训练')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--bias_wd", action="store_true", help="是否对bias参数进行weight decay")

    parser.add_argument("--comment", type=str, help="对本次实验的改动描述")
    parser.add_argument("--data_type", type=str, default="RADDet", help="数据集", choices=["RADDet", "CARRADA", "ALL"])
    parser.add_argument("--swanlab", action="store_true", default=False, help="是否使用swanlab进行记录")
    
    # evaluation parameters
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument('--all_mAP', action='store_true', default=True, help='EVALUATE ALL mAP')
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='decay factor for model weights moving average (default: 0.9999)')
    parser.add_argument('--use_film_metadata', action='store_true', default=False,
                        help='Whether to use FiLM metadata for conditioning')
    parser.add_argument('--bimamba_type', type=str, default='v4', choices=['v1', 'v2', 'v3', 'v4', 'none'],
                        help='Type of BiMamba to use if any')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\\n'))

    if args.swanlab and utils.is_main_process():
        swanlab.init(
            description=args.comment,
            config=args,
            logdir="./swanlab_dir",
            project="RAGM_finetune_RADDet",
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
            "main_finetune.py", 
            "engine_finetune_fpn.py",        
            "models/finetune_raddet.py", 
            "models/mamba_simple.py", 
            "models/selective_scan_interface.py", 
            "models/RADDet_finetune/RADDet_finetune_dataset.py",
            "models/RADDet_finetune/yolo_loss.py",
            "models/RADDet_finetune/yolo_head.py",
            "run_finetune1.sh",
        ]
        
        # 创建代码快照目录
        code_snapshot_dir = os.path.join(args.output_dir, "code_snapshots")
        os.makedirs(code_snapshot_dir, exist_ok=True)
        
        # 备份并记录代码文件
        for src_path in code_files:
            if os.path.exists(src_path):
                shutil.copy2(src_path, code_snapshot_dir)
                
        print(f"Model code snapshot saved to: {code_snapshot_dir}")

    # load dataset and config
    if args.eval:
        print("Evaluation mode, only load test dataset")
        dataset_train, dataset_val = Create_RADDet_Finetune_Dataset(args.RADDet_config_path, valdatatype='test', sensor_config_path=args.sensor_config_path)
    else:
        dataset_train, dataset_val = Create_RADDet_Finetune_Dataset(args.RADDet_config_path, valdatatype='test', sensor_config_path=args.sensor_config_path)

    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./models/RADDet_finetune/anchors.txt")
    config = loader.readConfig(config_file_name=args.RADDet_config_path)
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]
    num_classes = len(config_data["all_classes"])

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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
        persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存
        # prefetch_factor=args.prefetch_factor,
        # prefetch_factor=2
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存

    )

    # define model
    print(f"Creating model: {args.model}")
    model = models_raddet.__dict__[args.model](
        config_data=config_data,
        config_model=config_model,
        anchor_boxes=anchor_boxes,
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
            "yolo_head.bn1.weight", "yolo_head.bn1.bias",      
            "yolo_head.conv2.weight", "yolo_head.conv2.bias",
            # "patch_embed.proj.weight", "patch_embed.proj.bias",     # for  2 channels input
        ]
        for k in head_keys: 
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        utils.interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # print("--- Current Model State Dict Keys ---")
    # for k in model.state_dict().keys():
    #     print(k)
   
        
    if args.finetune and args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        # print("\n--- Checkpoint Model State Dict Keys ---")
        # if 'model' in checkpoint:
        #     for k in checkpoint['model'].keys():
        #         print(k)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
                
        utils.interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_ema = None
    if args.ema_decay > 0:
        model_ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
        print(f"Using Model EMA with decay = {args.ema_decay}")

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    if args.lr is None:
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

    param_groups = utils.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=args.bias_wd)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()
    criterion = RadDetLoss(
        input_size=torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device),
        focal_loss_iou_threshold=config_train["focal_loss_iou_threshold"]
    )

    utils.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    if args.eval:
        from models.RADDet_finetune.evaluate_mAP import test_mAP
        with model_ema.average_parameters():
            test_mAP(
                config_model=config_model,
                config_evaluate=config_evaluate,
                config_data=config_data,
                model=model,
                device=device,
                test_loader=data_loader_val,
                num_classes=num_classes,
                anchor_boxes=anchor_boxes,
                output_dir=args.output_dir,
            )
        return
    

    print(f"Start training for {args.epochs} epochs")
    best_val_ap = 0.0
    best_model_path = None
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model=model,
            model_ema=model_ema,
            criterion=criterion, 
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args,
            config_model=config_model,
            anchor_boxes=anchor_boxes,
        )
        
        # validation
        print(f"epochs: {epoch}, start validation")
        with model_ema.average_parameters():
            test_stats = evaluate(
                model=model,
                criterion=criterion,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                config_model=config_model,
                anchor_boxes=anchor_boxes,
                num_classes=num_classes,
                args = args,
                )
        
        current_val_ap = test_stats["ap_all"]  # 获取当前验证指标
        print(f"Mean AP of the network on the {len(dataset_val)} data: {current_val_ap:.4f}%")
        print(f'Best AP: {max(best_val_ap, current_val_ap):.4f}%')     

        # save best
        if current_val_ap > best_val_ap:
            best_val_ap = current_val_ap
            utils.save_model(
                args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler,
                model_ema=model_ema,  # <--- 将 model_ema 传递进去
                filename='best_model.pth' 
            )
            print(f"New best model saved with AP: {best_val_ap:.4f}")
        
        # save model periodically
        if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
            utils.save_model(
                args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler,
                model_ema=model_ema, # <--- 将 model_ema 传递进去
                filename=f'checkpoint-{epoch}.pth'
            )
            
        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": round(v, 6) for k, v in train_stats.items()},  
            **{f"test_{k}": round(v, 6) for k, v in test_stats.items()},
            "n_parameters": n_parameters,
            "best_val_ap": best_val_ap,
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\\n")

        if args.swanlab and utils.is_main_process():
            swanlab.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.swanlab and utils.is_main_process():
        swanlab.finish()


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