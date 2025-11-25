#!/usr/bin/env python3
"""
engine_finetune_raddet_ddp.py

DDP training engine. Uses repo dataset/model/loss and creates DistributedSampler
for proper distributed training. Modified to expect data_root layout:

data_root/
  train/
    annotations_pickle/
    camera_images/
    ra_matrices_NPY/
  test/
    annotations_pickle/
    camera_images/
    ra_matrices_NPY/
"""
import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ---------- Utilities ----------
def is_main_process(rank):
    return rank == 0

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def gather_metrics_dict(metrics: dict, world_size: int):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            t = torch.tensor(v, device='cuda')
            rt = reduce_tensor(t, world_size).item()
            out[k] = rt
        else:
            out[k] = v
    return out

# ---------- Dynamic import helpers ----------
def import_repo_modules(repo_path: Path):
    sys.path.insert(0, str(repo_path.resolve()))
    imports = {}
    # dataset attempts
    try:
        from dataset.radar_dataset import RadarDataset as RADDataset
        imports['dataset'] = RADDataset
    except Exception:
        try:
            from dataset.radar_dataset import RADDetDataset as RADDataset
            imports['dataset'] = RADDataset
        except Exception as e:
            raise ImportError("Failed to import dataset.radar_dataset RadarDataset/RADDetDataset. Adapt engine imports.") from e
    # model attempts
    try:
        from model.model import RADDet as RADModel
        imports['model_class'] = RADModel
    except Exception:
        try:
            from model.model import build_model as build_model_fn
            imports['model_builder'] = build_model_fn
        except Exception as e:
            raise ImportError("Failed to import model.model (RADDet or build_model). Adapt engine imports.") from e
    # loss attempts
    try:
        from model.yolo_loss import YOLOLoss as YOLOLoss
        imports['loss_class'] = YOLOLoss
    except Exception:
        try:
            from model.yolo_loss import yolo_loss as yolo_loss_fn
            imports['loss_fn'] = yolo_loss_fn
        except Exception as e:
            raise ImportError("Failed to import model.yolo_loss (YOLOLoss or yolo_loss). Adapt engine imports.") from e
    # evaluate attempts (optional)
    try:
        from evaluates.evaluate_mAP import evaluate_map
        imports['evaluate_map'] = evaluate_map
    except Exception:
        imports['evaluate_map'] = None
    return imports

# ---------- Core functions ----------
def create_dataloaders(dataset_cls, data_root, batch_size, num_workers, rank, world_size):
    """
    Build train/val dataloaders from actual directory layout:
      data_root/train/
      data_root/test/
    """
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'test')

    print(f"[INFO] Loading datasets from:")
    print(f"   Train dir: {train_dir}")
    print(f"   Val/Test dir: {val_dir}")

    # dataset construction attempts
    try:
        train_ds = dataset_cls(train_dir)
        val_ds = dataset_cls(val_dir)
    except TypeError:
        # try constructor signature dataset_cls(root, split=...)
        try:
            train_ds = dataset_cls(data_root, split='train')
            val_ds = dataset_cls(data_root, split='test')
        except Exception as e:
            raise RuntimeError("Cannot instantiate dataset. Adapt engine create_dataloaders.") from e

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader, train_sampler, val_sampler

def build_model_and_loss(imports, cfg, device):
    if 'model_class' in imports:
        Model = imports['model_class']
        try:
            model = Model(cfg.get('MODEL', {}))
        except Exception:
            model = Model()
    else:
        builder = imports.get('model_builder')
        if builder is None:
            raise RuntimeError("No model builder/class found")
        try:
            model = builder(cfg.get('MODEL', {}))
        except Exception:
            model = builder()
    if 'loss_class' in imports:
        LossClass = imports['loss_class']
        try:
            loss_fn = LossClass(cfg)
        except Exception:
            loss_fn = LossClass()
    elif 'loss_fn' in imports:
        loss_fn = imports['loss_fn']
    else:
        raise RuntimeError("No loss callable found")
    model.to(device)
    return model, loss_fn

def train_one_epoch_ddp(model, loss_fn, optimizer, loader, device, epoch, rank, world_size, print_freq=50):
    model.train()
    total_loss = 0.0
    n = 0
    loader.sampler.set_epoch(epoch)
    for i, sample in enumerate(loader):
        # unpack sample (common patterns)
        if isinstance(sample, dict):
            # try several common keys; prefer 'ra' or 'image' / 'img' for input
            images = sample.get('ra') or sample.get('image') or sample.get('images') or sample.get('img')
            targets = sample.get('anno') or sample.get('target') or sample.get('targets') or sample.get('label')
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            images, targets = sample[0], sample[1]
        else:
            raise RuntimeError("Unknown dataset return format; adapt engine unpacking.")
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        l = loss.item() if hasattr(loss, 'item') else float(loss)
        total_loss += l
        n += 1
        if i % print_freq == 0 and is_main_process(rank):
            print(f"[Rank {rank}] Epoch {epoch} Iter {i} loss {l:.4f}")
    avg_loss = total_loss / max(1, n)
    t = torch.tensor(avg_loss, device=device)
    avg_loss_all = reduce_tensor(t, dist.get_world_size()).item()
    return avg_loss_all

def evaluate_ddp(model, loader, device, evaluate_map_fn, rank, world_size):
    model.eval()
    if evaluate_map_fn is not None:
        if is_main_process(rank):
            print("[MAIN] Running repository evaluate_map (may run distributed-aware).")
        metrics = evaluate_map_fn(model, loader, device=device)
        reduced = gather_metrics_dict(metrics, world_size)
        return reduced
    total = 0
    with torch.no_grad():
        for sample in loader:
            if isinstance(sample, dict):
                images = sample.get('ra') or sample.get('image') or sample.get('images') or sample.get('img')
            elif isinstance(sample, (list, tuple)) and len(sample) >= 1:
                images = sample[0]
            else:
                raise RuntimeError("Unknown dataset return format in eval; adapt.")
            total += images.shape[0]
    t = torch.tensor(total, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_all = t.item()
    return {'samples': total_all}

def save_checkpoint(state, out_dir: Path, epoch: int, is_best=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(state, ckpt)
    if is_best:
        best = out_dir / 'best.pth'
        torch.save(state, best)

def main_ddp(rank, world_size, local_rank, args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if is_main_process(rank):
        print(f"[MAIN] Engine starting on rank {rank}/{world_size} device {device}")
    repo = Path(args.raddet_repo)
    if not repo.exists():
        raise FileNotFoundError(f"RADDet repo not found at {repo}")
    imports = import_repo_modules(repo)
    cfg_path = Path(args.config) if args.config else (repo / 'config.json')
    cfg = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    # Create dataloaders using actual train/ test folders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(imports['dataset'], args.data_root, args.batch_size, args.num_workers, rank, world_size)
    model, loss_fn = build_model_and_loss(imports, cfg, device)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = Path(args.output_dir) / args.exp_name
    best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch_ddp(model, loss_fn, optimizer, train_loader, device, epoch, rank, world_size)
        metrics = evaluate_ddp(model, val_loader, device, imports.get('evaluate_map'), rank, world_size)
        t1 = time.time()
        if is_main_process(rank):
            print(f"[MAIN] Epoch {epoch} done. avg_loss(all)={avg_loss:.4f}, metrics={metrics}, time={(t1 - t0):.1f}s")
        state = {
            'epoch': epoch,
            'model_state': model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
        }
        if is_main_process(rank):
            cur_metric = metrics.get('mAP', metrics.get('samples', 0))
            is_best = cur_metric > best_metric
            if is_best:
                best_metric = cur_metric
            save_checkpoint(state, out_dir, epoch, is_best=is_best)
    if is_main_process(rank):
        print("[MAIN] Training finished. Outputs in:", out_dir)
