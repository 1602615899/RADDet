import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from engine_finetune_raddet_unified import train_one_epoch, validate_one_epoch
from models.RCRODNet import RCRODNet
from dataset_raddet_adapter import RADDetDataset


# ==========================================================
# 初始化分布式训练环境
# ==========================================================
def setup_distributed(local_rank, world_size):
    """初始化分布式通信环境 (仅 Linux 使用)"""
    if sys.platform.startswith("win"):
        print("⚠️ Windows 环境不支持 NCCL，跳过 DDP 初始化。")
        return False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    return True


# ==========================================================
# 销毁分布式环境
# ==========================================================
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ==========================================================
# 主训练函数（单卡或多卡）
# ==========================================================
def main_worker(local_rank, args):
    # 自动识别是否使用 DDP
    use_ddp = torch.cuda.is_available() and torch.cuda.device_count() > 1 and not sys.platform.startswith("win")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if use_ddp:
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"✅ [DDP] Using GPU {local_rank}/{torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
    else:
        print("⚠️ [Single GPU / CPU] Running without DDP.")
        local_rank = 0

    # ===================== 数据集加载 =====================
    train_dataset = RADDetDataset(
        root=args.train_data,
        split='train'
    )
    val_dataset = RADDetDataset(
        root=args.val_data,
        split='val'
    )

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ===================== 模型定义 =====================
    model = RCRODNet(in_channels=4, num_classes=args.num_classes)
    model = model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ===================== 优化器定义 =====================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ===================== 训练/验证循环 =====================
    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        validate_one_epoch(model, val_loader, device, epoch, args)

        scheduler.step()

        if local_rank == 0:
            ckpt_path = os.path.join(args.save_dir, f"raddet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ 模型已保存至 {ckpt_path}")

    cleanup_distributed()


# ==========================================================
# 主入口
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Finetune RADDet (Single/Distributed)")
    parser.add_argument('--train_data', type=str, default="datasets/RADDet/train")
    parser.add_argument('--val_data', type=str, default="datasets/RADDet/val")
    parser.add_argument('--save_dir', type=str, default="checkpoints/raddet_finetune")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.device_count() > 1 and not sys.platform.startswith("win"):
        print(f"🌍 启动 DDP 模式，共 {args.world_size} 张 GPU。")
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        print("🖥️ 启动单 GPU / CPU 模式。")
        main_worker(local_rank=0, args=args)


if __name__ == "__main__":
    main()
