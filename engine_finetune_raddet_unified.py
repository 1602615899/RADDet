# #!/usr/bin/env python3
# """
# engine_finetune_raddet_unified.py

# Unified engine that runs either:
#  - Single-GPU / CPU training (Windows-friendly) OR
#  - Multi-GPU DDP training on Linux (torchrun / launch)

# Exports:
#  - train_one_epoch(model, dataloader, optimizer, device, epoch, args)
#  - validate_one_epoch(model, dataloader, device, epoch, args)
#  - main_ddp(rank, world_size, local_rank, args)   # used by DDP launcher
#  - run_single_process(args)                       # convenience runner for local debug

# Usage:
#  - For quick local debug (Windows): import and call run_single_process(args)
#  - For normal main script, keep using your main which imports train_one_epoch/validate_one_epoch
#  - For DDP: use torchrun --nproc_per_node=N main_finetune_raddet_ddp.py ...
# """

# import os
# import sys
# import json
# import time
# from pathlib import Path
# from typing import Tuple, Dict, Any

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, DistributedSampler
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# # --------- Utilities & Dynamic Import Helpers ---------
# def is_windows() -> bool:
#     return sys.platform.startswith("win")

# def print_if_main(rank, *args_, **kwargs):
#     if rank == 0:
#         print(*args_, **kwargs)

# def import_repo_modules(repo_path: Path) -> dict:
#     """
#     Try common import names from RADDet_Pytorch-main.
#     Returns dict with keys: dataset_class, model_class/model_builder, loss_class/loss_fn, evaluate_map (optional)
#     """
#     sys.path.insert(0, str(repo_path.resolve()))
#     imports = {}

#     # dataset
#     try:
#         from dataset.radar_dataset import RadarDataset as DatasetClass
#         imports['dataset_class'] = DatasetClass
#     except Exception:
#         try:
#             from dataset.radar_dataset import RADDetDataset as DatasetClass
#             imports['dataset_class'] = DatasetClass
#         except Exception as e:
#             raise ImportError("Failed to import dataset.radar_dataset (tried RadarDataset/RADDetDataset). "
#                               "Open RADDet_Pytorch-main/dataset to find the correct class name.") from e

#     # model
#     try:
#         from model.model import RADDet as ModelClass
#         imports['model_class'] = ModelClass
#     except Exception:
#         try:
#             from model.model import build_model as build_model_fn
#             imports['model_builder'] = build_model_fn
#         except Exception as e:
#             raise ImportError("Failed to import model.model (tried RADDet or build_model). "
#                               "Open RADDet_Pytorch-main/model to find correct entrypoint.") from e

#     # loss
#     try:
#         from model.yolo_loss import YOLOLoss as LossClass
#         imports['loss_class'] = LossClass
#     except Exception:
#         try:
#             from model.yolo_loss import yolo_loss as loss_fn
#             imports['loss_fn'] = loss_fn
#         except Exception as e:
#             raise ImportError("Failed to import model.yolo_loss (tried YOLOLoss/yolo_loss). "
#                               "Open RADDet_Pytorch-main/model/yolo_loss.py to find correct callable.") from e

#     # evaluate (optional)
#     try:
#         from evaluates.evaluate_mAP import evaluate_map
#         imports['evaluate_map'] = evaluate_map
#     except Exception:
#         imports['evaluate_map'] = None

#     return imports

# # --------- DataLoader builder (uses data_root/train and data_root/test) ---------
# def create_dataloaders(dataset_cls, data_root: str, batch_size: int, num_workers: int, rank: int = 0, world_size: int = 1
#                        ) -> Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
#     """
#     Attempt to construct dataset with common signatures:
#       - dataset_cls(root_dir)  OR
#       - dataset_cls(root_dir, split='train')
#       - If dataset returns dict in __getitem__, engine handles it.
#     """
#     train_dir = os.path.join(data_root, 'train')
#     val_dir = os.path.join(data_root, 'test')

#     if not os.path.isdir(train_dir):
#         raise FileNotFoundError(f"Train directory not found: {train_dir}")
#     if not os.path.isdir(val_dir):
#         raise FileNotFoundError(f"Val/Test directory not found: {val_dir}")

#     # instantiate dataset
#     try:
#         train_ds = dataset_cls(train_dir)
#         val_ds = dataset_cls(val_dir)
#     except TypeError:
#         # try signature (root, split=...)
#         try:
#             train_ds = dataset_cls(data_root, split='train')
#             val_ds = dataset_cls(data_root, split='test')
#         except Exception as e:
#             raise RuntimeError("Cannot instantiate dataset with tried signatures. "
#                                "Please adapt engine create_dataloaders to your dataset constructor.") from e

#     # samplers
#     if world_size > 1:
#         train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
#         val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
#     else:
#         train_sampler = None
#         val_sampler = None

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
#                               num_workers=num_workers, pin_memory=True, sampler=train_sampler)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
#                             num_workers=num_workers, pin_memory=True, sampler=val_sampler)
#     return train_loader, val_loader, train_sampler, val_sampler

# # --------- Model & Loss builder ---------
# def build_model_and_loss(imports: dict, cfg: dict, device: torch.device):
#     # model
#     if 'model_class' in imports:
#         Model = imports['model_class']
#         try:
#             model = Model(cfg.get('MODEL', {}))
#         except Exception:
#             model = Model()
#     elif 'model_builder' in imports:
#         builder = imports['model_builder']
#         try:
#             model = builder(cfg.get('MODEL', {}))
#         except Exception:
#             model = builder()
#     else:
#         raise RuntimeError("No model callable found in imports")

#     # loss
#     if 'loss_class' in imports:
#         LossClass = imports['loss_class']
#         try:
#             loss_fn = LossClass(cfg)
#         except Exception:
#             loss_fn = LossClass()
#     elif 'loss_fn' in imports:
#         loss_fn = imports['loss_fn']
#     else:
#         raise RuntimeError("No loss callable found in imports")

#     model.to(device)
#     return model, loss_fn

# # --------- Single-process training/validation (for Windows/local debug) ---------
# def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, args: Any, loss_fn=None, print_freq=20) -> float:
#     model.train()
#     if loss_fn is None:
#         # fallback generic
#         criterion = nn.CrossEntropyLoss()
#         def compute_loss(outputs, targets):
#             return criterion(outputs, targets)
#     else:
#         # loss_fn might be a class or function; handle both
#         if callable(loss_fn):
#             def compute_loss(outputs, targets):
#                 return loss_fn(outputs, targets)
#         else:
#             def compute_loss(outputs, targets):
#                 return loss_fn(outputs, targets)

#     running_loss = 0.0
#     n = 0
#     t0 = time.time()
#     for i, sample in enumerate(dataloader):
#         # unpack sample
#         if isinstance(sample, dict):
#             images = sample.get('ra') or sample.get('image') or sample.get('images') or sample.get('img')
#             targets = sample.get('anno') or sample.get('target') or sample.get('targets') or sample.get('label')
#         elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
#             images, targets = sample[0], sample[1]
#         else:
#             raise RuntimeError("Dataset __getitem__ returns unexpected format. Adapt train_one_epoch unpacking.")

#         images = images.to(device)
#         # targets left as-is (loss should handle)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = compute_loss(outputs, targets)
#         # if loss is a dict (some implementations return dict), sum or use 'loss'
#         if isinstance(loss, dict):
#             l = loss.get('loss') or sum(loss.values())
#             loss_val = l
#         else:
#             loss_val = loss
#         loss_val.backward()
#         optimizer.step()

#         l_scalar = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
#         running_loss += l_scalar
#         n += 1
#         if i % print_freq == 0:
#             print(f"[Train] Iter {i} loss {l_scalar:.4f}")

#     avg_loss = running_loss / max(1, n)
#     t1 = time.time()
#     print(f"[Train] Epoch {epoch} finished. avg_loss={avg_loss:.4f}. time={t1 - t0:.1f}s")
#     return avg_loss

# @torch.no_grad()
# def validate_one_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int, args: Any,
#                        loss_fn=None, print_freq=50) -> Dict[str, float]:
#     model.eval()
#     if loss_fn is None:
#         criterion = nn.CrossEntropyLoss()
#         def compute_loss(outputs, targets):
#             return criterion(outputs, targets)
#     else:
#         if callable(loss_fn):
#             def compute_loss(outputs, targets):
#                 return loss_fn(outputs, targets)
#         else:
#             def compute_loss(outputs, targets):
#                 return loss_fn(outputs, targets)

#     running_loss = 0.0
#     total = 0
#     correct = 0
#     t0 = time.time()
#     for i, sample in enumerate(dataloader):
#         if isinstance(sample, dict):
#             images = sample.get('ra') or sample.get('image') or sample.get('images') or sample.get('img')
#             targets = sample.get('anno') or sample.get('target') or sample.get('targets') or sample.get('label')
#         elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
#             images, targets = sample[0], sample[1]
#         else:
#             raise RuntimeError("Dataset __getitem__ returns unexpected format. Adapt validate_one_epoch unpacking.")

#         images = images.to(device)
#         outputs = model(images)
#         loss = compute_loss(outputs, targets)
#         l_scalar = loss.item() if hasattr(loss, 'item') else float(loss)
#         running_loss += l_scalar
#         total += images.shape[0] if hasattr(images, 'shape') else 1
#         # best-effort accuracy for classification-like outputs (not used for detection default)
#         try:
#             preds = outputs.argmax(dim=1)
#             correct += (preds == targets).sum().item()
#         except Exception:
#             pass

#         if i % print_freq == 0:
#             print(f"[Val] Iter {i} loss {l_scalar:.4f}")

#     avg_loss = running_loss / max(1, total)
#     acc = (correct / total * 100) if total > 0 else 0.0
#     t1 = time.time()
#     print(f"[Val] Epoch {epoch} finished. avg_loss={avg_loss:.4f}, acc={acc:.2f}%, time={t1 - t0:.1f}s")
#     return {'val_loss': avg_loss, 'acc': acc}

# # --------- DDP helpers (reduce tensors / gather metrics) ---------
# def reduce_tensor(tensor: torch.Tensor, world_size: int):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= world_size
#     return rt

# def gather_metrics_dict(metrics: dict, world_size: int):
#     out = {}
#     for k, v in metrics.items():
#         if isinstance(v, (int, float)):
#             t = torch.tensor(v, device='cuda' if torch.cuda.is_available() else 'cpu')
#             rt = reduce_tensor(t, world_size).item()
#             out[k] = rt
#         else:
#             out[k] = v
#     return out

# # --------- DDP training / eval (wraps single-process functions + sync) ---------
# def train_one_epoch_ddp(model: nn.Module, loss_fn, optimizer, loader: DataLoader, device: torch.device,
#                         epoch: int, rank: int, world_size: int, args: Any, print_freq=50) -> float:
#     model.train()
#     loader.sampler.set_epoch(epoch)
#     total_loss = 0.0
#     n = 0
#     t0 = time.time()
#     for i, sample in enumerate(loader):
#         if isinstance(sample, dict):
#             images = sample.get('ra') or sample.get('image') or sample.get('images') or sample.get('img')
#             targets = sample.get('anno') or sample.get('target') or sample.get('targets') or sample.get('label')
#         elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
#             images, targets = sample[0], sample[1]
#         else:
#             raise RuntimeError("Dataset __getitem__ returns unexpected format. Adapt train_one_epoch_ddp unpacking.")
#         images = images.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         # loss_fn expected to be callable
#         loss = loss_fn(outputs, targets) if loss_fn is not None else nn.CrossEntropyLoss()(outputs, targets)
#         if isinstance(loss, dict):
#             loss_val = loss.get('loss') or sum(loss.values())
#         else:
#             loss_val = loss
#         loss_val.backward()
#         optimizer.step()
#         l_scalar = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
#         total_loss += l_scalar
#         n += 1
#         if i % print_freq == 0 and rank == 0:
#             print(f"[Rank {rank}] Train Iter {i} loss {l_scalar:.4f}")
#     avg = total_loss / max(1, n)
#     t = torch.tensor(avg, device=device)
#     avg_all = reduce_tensor(t, world_size).item()
#     t1 = time.time()
#     if rank == 0:
#         print(f"[Rank {rank}] Epoch {epoch} finished. avg_loss(all)={avg_all:.4f}. time={t1-t0:.1f}s")
#     return avg_all

# def evaluate_ddp(model: nn.Module, loader: DataLoader, device: torch.device, evaluate_map_fn, rank: int, world_size: int):
#     model.eval()
#     if evaluate_map_fn is not None:
#         if rank == 0:
#             print("[MAIN] Running repository evaluate_map...")
#         metrics = evaluate_map_fn(model, loader, device=device)
#         reduced = gather_metrics_dict(metrics, world_size)
#         return reduced
#     total = 0
#     with torch.no_grad():
#         for sample in loader:
#             if isinstance(sample, dict):
#                 images = sample.get('ra') or sample.get('image') or sample.get('images') or sample.get('img')
#             elif isinstance(sample, (list, tuple)) and len(sample) >= 1:
#                 images = sample[0]
#             else:
#                 raise RuntimeError("Dataset __getitem__ returns unexpected format in eval.")
#             total += images.shape[0]
#     t = torch.tensor(total, device=device)
#     dist.all_reduce(t, op=dist.ReduceOp.SUM)
#     total_all = t.item()
#     return {'samples': total_all}

# # --------- Entry for DDP launcher (called by main) ---------
# def main_ddp(rank: int, world_size: int, local_rank: int, args):
#     """
#     rank: global rank
#     world_size: total world size
#     local_rank: local GPU index for this process
#     args: namespace with fields:
#       - raddet_repo, data_root, config (optional), output_dir, exp_name,
#       - epochs, batch_size, num_workers, lr
#     """
#     torch.backends.cudnn.benchmark = True
#     device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
#     if rank == 0:
#         print(f"[MAIN] DDP engine started rank {rank}/{world_size} device {device}")

#     repo = Path(args.raddet_repo)
#     if not repo.exists():
#         raise FileNotFoundError(f"RADDet repo not found at {repo}")
#     imports = import_repo_modules(repo)

#     # load config if exists
#     cfg = {}
#     cfg_path = Path(args.config) if getattr(args, 'config', None) else (repo / 'config.json')
#     if cfg_path.exists():
#         try:
#             with open(cfg_path, 'r') as f:
#                 cfg = json.load(f)
#         except Exception:
#             cfg = {}

#     # dataloaders
#     train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
#         imports['dataset_class'], args.data_root, args.batch_size, args.num_workers, rank, world_size
#     )

#     # model & loss
#     model, loss_fn = build_model_and_loss(imports, cfg, device)
#     model = model.to(device)
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     out_dir = Path(args.output_dir) / args.exp_name
#     best_metric = -float('inf')

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         avg_loss = train_one_epoch_ddp(model, loss_fn, optimizer, train_loader, device, epoch, rank, world_size, args)
#         metrics = evaluate_ddp(model, val_loader, device, imports.get('evaluate_map'), rank, world_size)
#         t1 = time.time()
#         if rank == 0:
#             print(f"[MAIN] Epoch {epoch} done. avg_loss(all)={avg_loss:.4f}, metrics={metrics}, time={(t1 - t0):.1f}s")
#         state = {
#             'epoch': epoch,
#             'model_state': model.module.state_dict(),
#             'optimizer_state': optimizer.state_dict(),
#             'metrics': metrics,
#         }
#         if rank == 0:
#             cur_metric = metrics.get('mAP', metrics.get('samples', 0))
#             is_best = cur_metric > best_metric
#             if is_best:
#                 best_metric = cur_metric
#             out_dir.mkdir(parents=True, exist_ok=True)
#             ckpt = out_dir / f'checkpoint_epoch_{epoch}.pth'
#             torch.save(state, ckpt)
#             if is_best:
#                 torch.save(state, out_dir / 'best.pth')
#     if rank == 0:
#         print("[MAIN] DDP training finished. Outputs in:", out_dir)

# # --------- Convenience runner for local single-process debug ---------
# def run_single_process(args):
#     """
#     args should have: raddet_repo, data_root, config (opt), output_dir, exp_name, epochs, batch_size, num_workers, lr
#     This runs single-process single-GPU (or CPU) training loop using train_one_epoch / validate_one_epoch.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"[LOCAL] Running single-process on device {device}")

#     repo = Path(args.raddet_repo)
#     if not repo.exists():
#         raise FileNotFoundError(f"RADDet repo not found at {repo}")
#     imports = import_repo_modules(repo)

#     cfg = {}
#     cfg_path = Path(args.config) if getattr(args, 'config', None) else (repo / 'config.json')
#     if cfg_path.exists():
#         try:
#             with open(cfg_path, 'r') as f:
#                 cfg = json.load(f)
#         except Exception:
#             cfg = {}

#     train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
#         imports['dataset_class'], args.data_root, args.batch_size, args.num_workers, rank=0, world_size=1
#     )

#     model, loss_fn = build_model_and_loss(imports, cfg, device)
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     out_dir = Path(args.output_dir) / args.exp_name
#     best_metric = -float('inf')
#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args, loss_fn)
#         metrics = validate_one_epoch(model, val_loader, device, epoch, args, loss_fn)
#         t1 = time.time()
#         print(f"[LOCAL] Epoch {epoch} done. avg_loss={avg_loss:.4f}, metrics={metrics}, time={(t1 - t0):.1f}s")
#         state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'metrics': metrics}
#         cur_metric = metrics.get('acc', metrics.get('val_loss', float('inf') * -1))
#         # save checkpoint
#         out_dir.mkdir(parents=True, exist_ok=True)
#         ckpt = out_dir / f'checkpoint_epoch_{epoch}.pth'
#         torch.save(state, ckpt)
#         if cur_metric > best_metric:
#             best_metric = cur_metric
#             torch.save(state, out_dir / 'best.pth')
#     print("[LOCAL] Training finished. Outputs in:", out_dir)

# # If this file run directly, provide a small CLI to debug
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser('engine_finetune_raddet_unified CLI (debug)')
#     parser.add_argument('--raddet_repo', type=str, default='./RADDet_Pytorch-main')
#     parser.add_argument('--data_root', type=str, default=r"C:\Users\DN\Desktop\RADdet_demo")
#     parser.add_argument('--config', type=str, default=None)
#     parser.add_argument('--output_dir', type=str, default='./outputs/raddet_unified')
#     parser.add_argument('--exp_name', type=str, default='raddet_unified')
#     parser.add_argument('--epochs', type=int, default=2)
#     parser.add_argument('--batch_size', type=int, default=4)
#     parser.add_argument('--num_workers', type=int, default=2)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--mode', choices=['local', 'ddp'], default='local', help='local for single-process; ddp for distributed (not used here)')
#     args = parser.parse_args()

#     if args.mode == 'local' or is_windows():
#         run_single_process(args)
#     else:
#         # For ddp mode you should launch per-process environment (torchrun / launch). This is helper only.
#         raise RuntimeError("To run DDP, use torchrun to spawn processes and call main_ddp from launcher.")


import os
import sys
import torch
import argparse
import importlib
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

# ================================
# 🔧 自动导入 RADDet 仓库模块
# ================================
def import_repo_modules(repo_path):
    repo_abs = os.path.abspath(repo_path)
    sys.path.insert(0, repo_abs)

    dataset_module = None
    DatasetClass = None

    try:
        from dataset.radar_dataset import RadarDataset as DatasetClass
        print("✅ Imported RadarDataset from dataset.radar_dataset")
    except ImportError:
        try:
            from dataset.radar_dataset import RADDetDataset as DatasetClass
            print("✅ Imported RADDetDataset from dataset.radar_dataset")
        except ImportError:
            try:
                from dataset.radar_dataset import RararDataset as DatasetClass
                print("✅ Imported RararDataset from dataset.radar_dataset")
            except ImportError:
                raise ImportError(
                    "❌ Failed to import dataset.radar_dataset.\n"
                    "Tried: RadarDataset, RADDetDataset, RararDataset.\n"
                    "Please check dataset/radar_dataset.py for correct class name."
                )

    return {"DatasetClass": DatasetClass}


# ================================
# 🧠 模型训练/验证循环
# ================================
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        inputs, labels, boxes = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"[Epoch {epoch}] Step {step}/{len(dataloader)} Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch} Training finished | Avg Loss: {avg_loss:.6f}")
    return avg_loss


def validate_one_epoch(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, boxes = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch} Validation finished | Avg Loss: {avg_loss:.6f}")
    return avg_loss


# ================================
# ⚙️ 单GPU训练逻辑
# ================================
def run_single_process(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOCAL] Running single-process on device {device}")

    # 🔹 自动导入 RADDet 模块
    imports = import_repo_modules(args.repo)
    DatasetClass = imports["DatasetClass"]

    # 🔹 载入数据集
    dataset = DatasetClass(
        config_data=args.config_data,
        config_train=args.config_train,
        config_model=args.config_model,
        headoutput_shape=[1, 8, 8, 8],  # 伪占位符
        anchors=[[1, 1, 1]],
        dType="train"
    )

    # 🔹 打印数据集信息
    print(f"✅ Loaded dataset: {len(dataset)} samples")
    sample = dataset[0]
    print(f"📦 Sample input shape: {sample[0].shape}")
    print(f"📦 Sample label shape: {sample[1].shape}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    # 🔹 简单模型占位符
    model = nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 2, kernel_size=3, padding=1)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 🔹 训练与验证
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, dataloader, optimizer, device, epoch)
        validate_one_epoch(model, dataloader, device, epoch)


# ================================
# 🧩 多GPU (DDP) 训练逻辑
# ================================
def run_ddp(args):
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"[DDP] Rank {local_rank} initialized on {device}")

    imports = import_repo_modules(args.repo)
    DatasetClass = imports["DatasetClass"]

    dataset = DatasetClass(
        config_data=args.config_data,
        config_train=args.config_train,
        config_model=args.config_model,
        headoutput_shape=[1, 8, 8, 8],
        anchors=[[1, 1, 1]],
        dType="train"
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2)

    model = nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 2, kernel_size=3, padding=1)
    ).to(device)

    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        train_one_epoch(model, dataloader, optimizer, device, epoch)
        validate_one_epoch(model, dataloader, device, epoch)

    dist.destroy_process_group()


# ================================
# 🚀 主程序入口
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="RADDet_Pytorch-main", help="Path to RADDet repo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--config_data", type=dict, default={"train_set_dir": "./data/train", "test_set_dir": "./data/test", "global_mean_log": 0, "global_variance_log": 1, "max_boxes_per_frame": 16, "all_classes": ["car", "pedestrian"]})
    parser.add_argument("--config_train", type=dict, default={"batch_size": 2, "epochs": 2, "if_validate": True})
    parser.add_argument("--config_model", type=dict, default={"input_shape": [2, 64, 64, 64]})
    parser.add_argument("--ddp", action="store_true", help="Enable DDP mode")
    args = parser.parse_args()

    if args.ddp:
        run_ddp(args)
    else:
        run_single_process(args)
