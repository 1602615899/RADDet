import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import swanlab
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import sys 
sys.path.append('./datasets/')
sys.path.append('./models/')
import utils 
from utils import NativeScalerWithGradNormCount as NativeScaler
from models.RADDet_finetune.RADDet_finetune_dataset_cart import Create_RADDet_Finetune_Dataset_cart
import models.RADDet_finetune.loader as loader
import models.radtr_yolo_cart as models_radtr_yolo_cart
from engine_finetune_cart import train_one_epoch, evaluate
import logging

def get_args_parser():
    parser = argparse.ArgumentParser('MAE cart pre-fituning', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints) \
                        梯度累积的次数，在显存受限的情况下，通过累积多个小批次的梯度，可以达到更大的等效批次大小，从而模拟大批次训练。')

    # Model parameters
    parser.add_argument('--model', default='RADTR_YOLO_tiny_cart', type=str, metavar='MODEL',
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
    parser.add_argument('--data_path', default='/mnt/SrvDataDisk/Datasets_Radar/RADDet/RADDet_author/train', type=str,
                        help='dataset path')
    parser.add_argument('--RADDet_config_path', default='./models/RADDet_finetune/config_cart.json', type=str,
                        help='RADDet_config_path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:1',     # 主gpu
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
    parser.add_argument("--sep_pos_embed", action="store_true", default=False)
    parser.add_argument("--attn_type", default="Normal", type=str)
    parser.add_argument("--decoder_attn_type", default="Normal", type=str)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--bias_wd", action="store_true", help="是否对bias参数进行weight decay")

    parser.add_argument("--comment", type=str, help="对本次实验的改动描述，或想探究的点")

    parser.add_argument("--data_type", type=str, default="RADDet", help="数据集", choices=["RADDet", "CARRADA", "ALL"])

    parser.add_argument("--swanlab", action="store_true", default=False, help="是否使用swanlab进行记录")
    
    # evaluation parameters
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    parser.add_argument('--drop_path_rate', type=float, default=0.00, help='drop_path_rate in model')

    return parser


def main(args):

    utils.init_distributed_mode(args)

    # print args
    # print(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.swanlab and utils.is_main_process():
        retries = 10
        for attempt in range(retries):
            success = False
            try:
                swanlab.init(project="mae_finetune", notes=args.comment, config=args, dir='./swanlab_dir')
                # wandb.init(settings=wandb.Settings(init_timeout=120))
                success = True
                break  # 如果成功，退出循环
            except Exception as e:
                print(f"尝试连接 swanlab 失败: {e}, 尝试次数: {attempt + 1}")
                time.sleep(5)
                

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

    # dataset_train = radar_datasets(args.data_type)

    # load dataset and config
    if args.data_type == "RADDet":
        if args.eval:
            print("Evaluation mode, only load test dataset")
            dataset_train, dataset_val = Create_RADDet_Finetune_Dataset_cart(args.RADDet_config_path, valdatatype='test')
        else:
            dataset_train, dataset_val = Create_RADDet_Finetune_Dataset_cart(args.RADDet_config_path)
        
        anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./models/RADDet_finetune/anchors_cartboxes.txt")
        config = loader.readConfig(config_file_name=args.RADDet_config_path)
        config_data = config["DATA"]
        config_radar = config["RADAR_CONFIGURATION"]
        config_model = config["MODEL"]
        config_train = config["TRAIN"]
        config_evaluate = config["EVALUATE"]
        num_classes = len(config_data["all_classes"])
    else:
        raise ValueError("Invalid data_type: %s" % args.data_type)

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
                dataset_val, num_replicas=num_tasks, rank=global_rank, drop_last=False, shuffle=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) # Samples elements sequentially, always in the same order.
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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
    # 传入args.model参数，返回相应的模型
    model = models_radtr_yolo_cart.__dict__[args.model](
        config_data=config_data,
        config_model=config_model,
        config_train=config_train,
        anchor_boxes=anchor_boxes,
        attn_type=args.attn_type,
        drop_path_rate=args.drop_path_rate,
    )   
    
    ## hiera
    # model = radar_hiera_yolo.__dict__[args.model](
    #     config_data=config_data,
    #     config_model=config_model,
    #     anchor_boxes=anchor_boxes,
    #     # drop_path_rate=args.drop_path_rate,
    #     stages=(2,3,8,3)
    # )   

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
                
        state_dict = model.state_dict()
        
        head_keys = [
            "yolo_head.conv.weight", "yolo_head.conv.bias",
            "yolo_head.bn.weight", "yolo_head.bn.bias",       # 这里用batch normalization可以吗？前面用的都是layer normalization
            "yolo_head.conv.weight", "yolo_head.conv.bias",
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

    utils.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    
    params_list = list(model_without_ddp.named_parameters())
    
    if args.eval:
        test_stats = evaluate(
            model=model,
            data_loader=data_loader_val,
            device=device,
            epoch=0,
            config_model=config_model,
            anchor_boxes=anchor_boxes,
            num_classes=num_classes,
            )
        
        # from models.RADDet_finetune.evaluate_mAP import test_mAP
        
        # test_mAP(
        #     config_model=config_model,
        #     config_evaluate=config_evaluate,
        #     config_data=config_data,
        #     model=model,
        #     device=device,
        #     test_loader=data_loader_val,
        #     num_classes=num_classes,
        #     anchor_boxes=anchor_boxes,
        #     output_dir=args.output_dir,
        # )
        
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    best_val_ap = 0.0
    best_model_path = None
    start_time = time.time()
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
            anchor_boxes=anchor_boxes,
        )
            
        # validation
        print(f"epochs: {epoch}, start validation")
        test_stats = evaluate(
            model=model,
            data_loader=data_loader_val,
            device=device,
            epoch=epoch,
            log_writer=log_writer,
            config_model=config_model,
            anchor_boxes=anchor_boxes,
            num_classes=num_classes,
            )
        
        current_val_ap = test_stats["ap_all"]  # 获取当前验证指标
        print(f"Mean AP of the network on the {len(dataset_val)} data: {current_val_ap:.4f}%")
        print(f'Best AP: {max(best_val_ap, current_val_ap):.4f}%')
            
        # save best
        if current_val_ap > best_val_ap:
            best_val_ap = current_val_ap
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
            print(f"New best model saved with AP: {best_val_ap:.4f}")
            
            
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
            
        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": f"{v:.6f}" for k, v in train_stats.items()},
            **{f"test_{k}": f"{v:.6f}" for k, v in test_stats.items()},
            "n_parameters": n_parameters,
            "best_val_ap": best_val_ap,
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.swanlab:
                swanlab.log(test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # print(torch.cuda.memory_allocated())

    if args.swanlab:
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