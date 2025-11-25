import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import torch
import torch.distributed as dist
from torch import inf
import math
import psutil
import logging
import torch.nn as nn
from timm.utils import get_state_dict

########################## DDP utils ########################################

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)



########################## Training utils ########################################

def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            # print("no decay: ", name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

from torch.cuda.amp import GradScaler
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        # self._scaler = torch.cuda.amp.GradScaler()    
        # self._scaler = torch.amp.GradScaler('cuda')
        self._scaler = GradScaler()
    #  `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

# In utils.py

def load_model_pretrain(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    统一加载检查点，适用于预训练和微调。
    *** 新版本：智能处理因模型结构变化导致的权重形状不匹配问题 ***
    """
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"==> Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # --- 智能加载模型权重 ---
            checkpoint_model = checkpoint.get('model', checkpoint) # 兼容旧格式
            model_state_dict = model_without_ddp.state_dict()
            
            # 1. 过滤掉检查点中形状不匹配的权重
            pretrained_dict = {}
            for k, v in checkpoint_model.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    print(f"==> Skipping loading layer '{k}' due to shape mismatch:")
                    print(f"    Checkpoint shape: {v.shape}, Model shape: {model_state_dict.get(k, 'N/A')}")
            
            # 2. 更新当前模型的 state_dict
            model_state_dict.update(pretrained_dict)
            
            # 3. 加载权重
            # 使用 strict=False，因为我们已经手动处理了不匹配的层
            # load_state_dict 会返回一个包含未加载和意外键的 named tuple
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(model_state_dict, strict=False)

            if missing_keys:
                print("\n==> Warning: The following keys were not found in the checkpoint and were initialized randomly:")
                print(missing_keys)
            if unexpected_keys:
                # 这个分支理论上不应该被触发，因为我们已经过滤了
                print("\n==> Warning: The following keys were in the checkpoint but not in the model:")
                print(unexpected_keys)

            print("\n==> Model weights loaded successfully (with expected mismatches handled).")
            
            # --- 加载优化器、scaler 和 epoch (逻辑不变) ---
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint['epoch'] + 1
                    if 'scaler' in checkpoint and loss_scaler is not None:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print(f"==> Resumed training from epoch {args.start_epoch}")
                except Exception as e:
                    print(f"==> Warning: Could not load optimizer state. Error: {e}")
                    print("==> Optimizer will start from scratch. This is normal if model architecture changed.")

            # --- 加载EMA模型 (逻辑不变) ---
            if model_ema is not None and 'model_ema' in checkpoint:
                try:
                    # 也为EMA模型进行智能加载
                    ema_state_dict = model_ema.state_dict()
                    checkpoint_ema = checkpoint.get('model_ema', {})
                    pretrained_ema_dict = {
                        k: v for k, v in checkpoint_ema.items() 
                        if k in ema_state_dict and v.shape == ema_state_dict[k].shape
                    }
                    ema_state_dict.update(pretrained_ema_dict)
                    model_ema.load_state_dict(ema_state_dict, strict=False)
                    print("==> Resuming EMA model state (with mismatch handling).")
                except Exception as e:
                    print(f"==> Warning: Could not load EMA model state. Error: {e}")
            
        else:
            print(f"==> No checkpoint found at '{args.resume}'")

def load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    统一加载检查点，适用于预训练和微调。
    该函数没有返回值，直接修改传入的对象。
    """
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"==> Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # 加载主模型
            # 使用 strict=False 可以更灵活地处理模型结构略有变化的情况
            msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            print(f"Model loading message: {msg}")
            
            # 加载优化器、scaler 和 epoch (仅在非评估模式下)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint['epoch'] + 1
                    if 'scaler' in checkpoint and loss_scaler is not None:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print(f"==> Resumed training from epoch {args.start_epoch}")
                except ValueError:
                    print("==> Warning: Could not load optimizer state. This is normal if you are changing model architecture or optimizer type.")

            # 加载EMA模型
            if model_ema is not None and 'model_ema' in checkpoint:
                try:
                    print("==> Resuming EMA model state")
                    model_ema.load_state_dict(checkpoint['model_ema'])
                except Exception as e:
                    print(f"==> Warning: Could not load EMA model state. Error: {e}")
            
            # 不再加载或返回 best_val_ap

        else:
            print(f"==> No checkpoint found at '{args.resume}'")


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, filename='checkpoint.pth'):
    """
    统一保存检查点，适用于预训练和微调。
    - filename 参数允许我们指定保存的文件名，例如 'checkpoint-10.pth' 或 'best_model.pth'
    """
    # 确保loss_scaler不为None时才保存其状态
    scaler_state_dict = loss_scaler.state_dict() if loss_scaler is not None else None

    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': scaler_state_dict,
        'args': args,
    }
    
    if model_ema is not None:
        to_save['model_ema'] = model_ema.state_dict()

    filepath = Path(args.output_dir) / filename
    save_on_master(to_save, filepath)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_v2(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup and scale * epochs"""
    scale = 0.4
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    elif  epoch > scale*args.epochs:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - scale*args.epochs) / (args.epochs - scale*args.epochs)))
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        






def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
    

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total

# 这里目前用不上，但是后面得改，首先是height和width不一定相等
# 插值的时候可以直接用它这个吗，我的是3维，参考VideoMAE的实现，照着mae的实现看看内部结构怎么回事
# 这里还没有改，但是改的思路写在这了，得照着数据维度改
def interpolate_pos_embed(model, checkpoint_model):
    '''
    输入与预训练时的输入size不同时，需要进行插值
    这里的插值方式是bicubic
    '''
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # num_patches = model.RAD_transformer.patch_embed.num_patches     # 这里是两块，所以加上RAD_transformer
        # num_extra_tokens = model.RAD_transformer.pos_embed.shape[-2] - num_patches
        
        # height (== width) for the checkpoint position embedding   
        # 这里的orig_size的H和W不一定相等，应该改成R和A，而不是单独一个orig_size重复两次
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # 这里应该从B, L, Dim -> B, Doppler, R, A, Dim -> B, Dim, Doppler, R, A -> interpolate
            # 参考之前写的patchify和unpatchify，思路都差不多
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # 插值后，再恢复：B, Dim, Doppler, R, A -> B, Doppler, R, A, Dim -> B, L, Dim
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
        else:
            print("Position interpolate: no need")
            
            
# 继承自 sys.stdout，用于捕获 print 输出
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.level = log_level

    def write(self, message):
        if message != '\n':  # 忽略空行
            self.logger.log(self.level, message.strip())

    def flush(self):
        # 使得 flush 可用
        pass
    
    
class layernorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class layernorm_3d(nn.Module):
    """
    只适用于三维数据
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width, depth).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # x = self.weight[:, None, None] * x + self.bias[:, None, None]
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x