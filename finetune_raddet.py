import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, Tuple
import math
import numpy as np
import torch.nn.functional as F
from timm.models.vision_transformer import _load_weights
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_, DropPath
from einops import rearrange, repeat

from mamba_ssm.modules.mamba_simple import Mamba 
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# ================= FiLM 模块定义 =================
class FiLMLayer(nn.Module):
    """
    特征级线性调制层 (FiLM Layer)。
    该层根据一个外部的条件向量，生成缩放因子gamma和偏移因子beta，
    并将其应用于输入的特征图。
    """
    def __init__(self, condition_dim, feature_dim):
        """
        初始化FiLM层。
        Args:
            condition_dim (int): 条件向量的维度。
            feature_dim (int): 要调制的特征的维度。
        """
        super().__init__()
        # 一个简单的前馈网络，用于从条件向量生成gamma和beta
        self.generator = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.GELU(),
            nn.Linear(condition_dim * 2, feature_dim * 2) # 输出维度是特征维度的两倍 (gamma + beta)
        )

    def forward(self, x, cond_vector):
        """
        前向传播。
        Args:
            x (Tensor): 输入特征，形状为 [B, N, C] (Batch, Num_Tokens, Channels)。
            cond_vector (Tensor): 条件向量，形状为 [B, D] (Batch, Condition_Dim)。
        
        Returns:
            Tensor: 经过调制的特征，形状与x相同。
        """
        # 1. 生成 gamma 和 beta
        # generator的输出形状为 [B, C*2]
        gb = self.generator(cond_vector)
        
        # 将gb切分为gamma和beta
        gamma, beta = torch.chunk(gb, 2, dim=1) # 每个形状为 [B, C]
        
        # 2. 调整形状以进行广播
        # [B, C] -> [B, 1, C]，以便与 [B, N, C] 的x进行元素级操作
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # 3. 应用FiLM变换: x' = γ * x + β
        # 为了训练稳定性，通常初始化时让gamma接近1，beta接近0。
        # 这里通过加1来实现，使得在网络初始化时，FiLM层近似于一个恒等变换。
        return (gamma + 1) * x + beta

# ================= 元数据编码器 =================
class MetadataEncoder(nn.Module):
    """
    将雷达元数据编码为FiLM所需的条件向量。
    """
    def __init__(self, num_datasets, numerical_dim, output_dim):
        """
        初始化元数据编码器。
        Args:
            num_datasets (int): 数据集类型的数量 (例如: RADDet, CARRADA等)。
            numerical_dim (int): 数值型元数据的维度 (例如: [range_res, angle_res, vel_res])。
            output_dim (int): 输出条件向量的目标维度。
        """
        super().__init__()
        # 1. 类别型数据编码器
        self.dataset_embedding_dim = 32 # 为数据集ID分配的嵌入维度
        self.dataset_embedder = nn.Embedding(num_datasets, self.dataset_embedding_dim)

        # 2. 数值型数据编码器 (一个简单的线性层)
        self.numerical_proj_dim = 64
        self.numerical_projector = nn.Linear(numerical_dim, self.numerical_proj_dim)

        # 3. 融合层
        # 将编码后的类别型和数值型特征拼接后，通过一个MLP进行融合
        total_input_dim = self.dataset_embedding_dim + self.numerical_proj_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, dataset_ids, numerical_params):
        """
        前向传播。
        Args:
            dataset_ids (Tensor): 数据集ID，形状 [B]。
            numerical_params (Tensor): 数值型参数，形状 [B, numerical_dim]。

        Returns:
            Tensor: 输出的条件向量，形状 [B, output_dim]。
        """
        # 编码类别型数据
        dataset_emb = self.dataset_embedder(dataset_ids) # [B, 32]

        # 编码数值型数据
        numerical_proj = self.numerical_projector(numerical_params) # [B, 64]
        
        # 拼接
        combined_features = torch.cat([dataset_emb, numerical_proj], dim=1) # [B, 32+64]

        # 融合生成最终的条件向量
        cond_vector = self.fusion_mlp(combined_features) # [B, output_dim]
        
        return cond_vector

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 网格高度、宽度和深度的整数元组
    返回:
    pos_embed: [grid_depth*grid_height*grid_width, embed_dim] 或 [1+grid_depth*grid_height*grid_width, embed_dim] (带cls_token)
    """
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # 注意这里的w, h, d顺序很重要
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[1], grid_size[2], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 6 == 0

    # 使用1/3维度给grid_h，1/3给grid_w，1/3给grid_d
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D, D/3)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1) # (H*W*D, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 每个位置的输出维度
    pos: 要编码的位置列表：大小 (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), 外积

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_3d_physical_pos_embed_dynamic(embed_dim, physical_coords):
    """
    动态生成物理位置编码（完全向量化）
    physical_coords: [batch_size, num_tokens, 3] 物理坐标 (距离, 角度, 速度/时间)
    返回: [batch_size, num_tokens, embed_dim]
    """
    batch_size, num_tokens, _ = physical_coords.shape
    device = physical_coords.device
    
    # 确保维度能够均匀分配给三个坐标维度
    dim_per_coord = embed_dim // 3
    remaining_dims = embed_dim % 3
    
    pos_embed = torch.zeros(batch_size, num_tokens, embed_dim, device=device)
    
    start_dim = 0
    for j in range(3):  # 距离、角度、速度/时间
        # 为当前坐标分配维度（第一个坐标获得额外维度）
        current_dim = dim_per_coord + (1 if j < remaining_dims else 0)
        
        if current_dim == 0:
            continue
            
        end_dim = start_dim + current_dim
        
        # 生成频率参数（确保是偶数维度用于sin/cos对）
        if current_dim % 2 == 1:
            # 如果是奇数维度，使用current_dim-1来生成频率
            freq_dim = current_dim - 1
            use_extra_dim = True
        else:
            freq_dim = current_dim
            use_extra_dim = False
            
        if freq_dim > 0:
            omega = torch.arange(freq_dim // 2, dtype=torch.float32, device=device)
            omega /= freq_dim / 2.
            omega = 1. / 10000**omega  # (freq_dim/2,)
            
            coord_data = physical_coords[:, :, j]  # [batch_size, num_tokens]
            
            # 批量计算 (完全向量化)
            out = torch.einsum('bn,d->bnd', coord_data, omega)  # (batch_size, num_tokens, freq_dim/2)
            emb_sin = torch.sin(out)
            emb_cos = torch.cos(out)
            combined_emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (batch_size, num_tokens, freq_dim)
        else:
            combined_emb = torch.zeros(batch_size, num_tokens, 0, device=device)
        
        # 如果需要额外维度，添加一个简单的编码
        if use_extra_dim:
            extra_emb = physical_coords[:, :, j:j+1] * 0.01  # 简单的线性编码
            combined_emb = torch.cat([combined_emb, extra_emb], dim=-1)
        
        pos_embed[:, :, start_dim:end_dim] = combined_emb
        start_dim = end_dim
    
    return pos_embed  # [batch_size, num_tokens, embed_dim]

def generate_physical_coords_fully_vectorized(batch_params_tensor, grid_size, has_velocity_mask, device):
    """
    完全向量化的物理坐标生成（无任何Python循环）
    batch_params_tensor: [B, 4] 的张量, [range_res, angle_res, vel_res/time_res, is_velocity_flag]
    grid_size: (H, W, D)
    has_velocity_mask: [B] 布尔张量，标识每个样本是否有速度维度
    device: 计算设备
    返回: [batch_size, num_tokens, 3]
    """
    B = batch_params_tensor.shape[0]
    H, W, D = grid_size
    
    # 1. 创建索引网格 (h, w, d) - 扩展到 [B, H, W, D]
    h_indices = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, W, D)  # [B, H, W, D]
    w_indices = torch.arange(W, device=device).view(1, 1, W, 1).expand(B, H, W, D)  # [B, H, W, D]
    d_indices = torch.arange(D, device=device).view(1, 1, 1, D).expand(B, H, W, D)  # [B, H, W, D]
    
    # 2. 准备物理分辨率张量以进行广播
    range_res = batch_params_tensor[:, 0].view(B, 1, 1, 1)  # [B, 1, 1, 1]
    angle_res = batch_params_tensor[:, 1].view(B, 1, 1, 1)  # [B, 1, 1, 1]
    vel_or_time_res = batch_params_tensor[:, 2].view(B, 1, 1, 1)  # [B, 1, 1, 1]
    
    # 3. 使用广播机制一次性计算所有坐标 - 完全向量化
    distances = h_indices * range_res  # [B, H, W, D]
    angles = (w_indices - W / 2) * angle_res  # [B, H, W, D]
    
    # 处理速度/时间维度的条件广播
    velocities = torch.where(
        has_velocity_mask.view(B, 1, 1, 1),
        (d_indices - D / 2) * vel_or_time_res,  # 速度模式
        d_indices * vel_or_time_res  # 时间模式 (CRUW)
    )  # [B, H, W, D]
    
    # 4. 堆叠并展平 - 完全向量化
    coords = torch.stack([distances, angles, velocities], dim=-1).view(B, -1, 3)
    
    # 5. 批处理归一化 (关键) - 完全向量化
    max_vals = coords.max(dim=1, keepdim=True)[0]  # [B, 1, 3]
    min_vals = coords.min(dim=1, keepdim=True)[0]  # [B, 1, 3]
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
    
    coords_normalized = (coords - min_vals) / range_vals
    
    return coords_normalized

def prepare_batch_params_tensor(batch_params, device):
    """
    [修改后的版本]
    将张量字典格式的参数直接转换为所需张量（完全向量化）。
    batch_params: 每个键都对应一个张量，形状为 [B]。
    返回: [B, 4] 张量和 [B] 布尔掩码。
    """
    # 从字典中直接获取整批的张量
    range_res = batch_params['range_resolution'].to(device)
    angle_res = batch_params['angular_resolution'].to(device)
    
    B = range_res.shape[0]
    default_vel_res = torch.full((B,), 0.05, device=device)
    vel_or_time_res = batch_params.get('velocity_resolution', batch_params.get('time_res', default_vel_res)).to(device)
    
    default_has_vel = torch.zeros(B, dtype=torch.bool, device=device)
    has_velocity = batch_params.get('has_velocity', default_has_vel).to(device)

    # 创建速度掩码和速度标记
    has_velocity_mask = has_velocity.bool() 
    is_velocity_flag = has_velocity_mask.float()

    # 使用 torch.stack 将所有参数张量堆叠起来
    batch_params_tensor = torch.stack([
        range_res.float(),
        angle_res.float(),
        vel_or_time_res.float(),
        is_velocity_flag
    ], dim=1)

    return batch_params_tensor, has_velocity_mask


class singleLayerHead3D(nn.Module):
    def __init__(self, num_anchors, num_class, channel, **kwargs):
        super(singleLayerHead3D, self).__init__()
        self.num_anchor = num_anchors
        self.num_class = num_class
        self.channel = channel

        final_output_channels = int(self.num_anchor * (num_class + 7))  # 78

        self.conv1 = nn.Conv3d(in_channels=self.channel,
                               out_channels=self.channel*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm3d(self.channel*2)

        self.conv2 = nn.Conv3d(in_channels=self.channel*2,
                               out_channels=final_output_channels,
                               kernel_size=1,
                               stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_feature):
        x = self.relu(self.bn1(self.conv1(input_feature)))
        x = self.conv2(x)
        return x

# ========= FPN相关辅助类 (来自radtr_yolo.py) =========
class ConvBNAct3D(nn.Sequential):
    """Standard Conv3D -> BatchNorm3d -> Activation block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.SiLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            activation(inplace=True)
        )

class ConvDownsample3D(ConvBNAct3D):
    """ConvBNAct3D with stride=2 for downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(in_channels, out_channels, kernel_size, stride=2)

class PANetFusion3D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.downsample = ConvDownsample3D(embed_dim, embed_dim, kernel_size=3)

        self.lat_f0 = nn.Conv3d(embed_dim, embed_dim//4, kernel_size=1)
        self.lat_d1 = nn.Conv3d(embed_dim, embed_dim//2, kernel_size=1)
        self.lat_d2 = nn.Conv3d(embed_dim, embed_dim//2, kernel_size=1)

        self.fuse_td1 = ConvBNAct3D(embed_dim//2, embed_dim//4, kernel_size=3)
        self.fuse_td0 = ConvBNAct3D(embed_dim//4, embed_dim//4, kernel_size=3)

        self.downsample_bu1 = ConvDownsample3D(embed_dim//4, embed_dim//4, kernel_size=3)
        self.downsample_bu2 = ConvDownsample3D(embed_dim//2, embed_dim//2, kernel_size=3)

        self.fuse_bu1 = ConvBNAct3D(embed_dim//4, embed_dim//2, kernel_size=3)
        self.fuse_bu2 = ConvBNAct3D(embed_dim//2, embed_dim, kernel_size=3)

    def forward(self, features):
        # Input: list [f0, f1, f2], all initially [B, C, D, H, W]
        f0, f1, f2 = features

        # --- Create Multi-Scale Features ---
        # Upsample f0 to create d0
        d0 = self.upsample(f0)
        # Downsample f2 to create d2
        d2 = self.downsample(f2)

        # --- Top-Down Path ---
        # Apply lateral connections
        lat0 = self.lat_f0(d0)
        lat1 = self.lat_d1(f1)
        lat2 = self.lat_d2(d2)

        # P2 (from smallest feature map d2)
        p2 = lat2 # Feature map at the smallest scale
        # P1 = Upsample(P2) + Lateral(f1) -> Refine
        p1_sum = self.upsample(p2) + lat1
        p1 = self.fuse_td1(p1_sum)
        # P0 = Upsample(P1) + Lateral(d0) -> Refine
        p0_sum = self.upsample(p1) + lat0
        p0 = self.fuse_td0(p0_sum)
        # --- Bottom-Up Path ---
        # N0 is the same as P0 for the highest resolution
        n0 = p0

        # N1 = Downsample(N0) + P1 -> Refine
        n1_sum = self.downsample_bu1(n0) + p1 # Reuse p1 from top-down
        n1 = self.fuse_bu1(n1_sum)
        
        # N2 = Downsample(N1) + P2 -> Refine
        n2_sum = self.downsample_bu2(n1) + p2 # Reuse p2 from top-down
        n2 = self.fuse_bu2(n2_sum)

        return n0, n1, n2 # Shape should be [B, C, D, H, W] (same as f0)
    
class PatchEmbed3D(nn.Module):
    """RAD tensor to 3D(RAD) Patch Embedding"""
    def __init__(self, input_size=(256, 256, 64), patch_size=(16, 16, 16), in_chans=1, embed_dim=192):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        
        # 动态计算网格大小
        self.grid_size = (
            input_size[0] // patch_size[0],  # H
            input_size[1] // patch_size[1],  # W
            input_size[2] // patch_size[2]   # D
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # 输入x形状: [B, C, H, W, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class ARBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, drop_path=0.,
        fused_add_norm=False, residual_in_fp32=False,
        use_film=False, condition_dim=None
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.mlp = SwiGLU(dim, dim*4*2//3, subln=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.use_film = use_film
        if self.use_film:
            assert condition_dim is not None, "condition_dim must be provided if use_film is True"
            self.film_layer = FiLMLayer(condition_dim=condition_dim, feature_dim=dim)


    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, cond_vector: Optional[Tensor] = None):
        normed_states = self.norm1(hidden_states)
        
        if self.use_film and cond_vector is not None:
            modulated_states = self.film_layer(normed_states, cond_vector)
        else:
            modulated_states = normed_states

        mixer_out = self.mixer(
            modulated_states, 
            inference_params=inference_params
        )

        hidden_states = hidden_states + self.drop_path(mixer_out)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
    use_film=False,
    condition_dim=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, expand=1, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = ARBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        use_film=use_film,
        condition_dim=condition_dim,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# ========= 主要微调模型 =========
class RadDet(nn.Module):

    def __init__(self, 
                 config_data,
                 config_model,
                 anchor_boxes,
                 embed_dim=192,
                 depth=12,
                 ssm_cfg=None, 
                 drop_path_rate=0,
                 modality_embed_dim=64, # ### FIX 2: 添加 modality_embed_dim 参数
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_film_metadata=True,
                 condition_dim=12,
                 **kwargs):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # 保存模型配置
        self.config_data = config_data
        self.config_model = config_model
        self.anchor_boxes = anchor_boxes
        self.embed_dim = embed_dim  # 保存为实例变量
        self.depth = depth          # 保存为实例变量
        
        # 模型参数 - 支持配置
        input_size = (256, 256, 64)
        patch_size = (8, 8, 8)  # 统一为(16, 16, 16)，与预训练模型保持一致
        # embed_dim 和 depth 现在作为参数传入
        
        print(f"微调模型初始化 - embed_dim: {embed_dim}, depth: {depth}")
        
        # Patch Embedding
        self.patch_embed = PatchEmbed3D(
            input_size=input_size,
            patch_size=patch_size, 
            in_chans=1,  # RADDet使用1个通道
            embed_dim=embed_dim
        )
        # self.pos_drop = nn.Dropout(p=pos_drop_rate) # ### FIX 1: 初始化 pos_drop
        # 动态计算网格大小和分组大小
        self.grid_size = (
            input_size[0] // patch_size[0],
            input_size[1] // patch_size[1],
            input_size[2] // patch_size[2]
        )
        self.group_size = (
            min(4, self.grid_size[0]),
            min(4, self.grid_size[1]),
            min(4, self.grid_size[2])
        )
        print(f"网格大小: {self.grid_size}, 分组大小: {self.group_size}")
        self.register_buffer("grid_tensor", torch.tensor(self.grid_size, dtype=torch.long))
        self.register_buffer("group_tensor", torch.tensor(self.group_size, dtype=torch.long))
        
        # 模态类型嵌入 (保持分辨率信息)
        self.modality_embedding = nn.Embedding(2, modality_embed_dim)
        
        # 调整物理位置编码维度
        self.physical_pos_embed_dim = embed_dim - modality_embed_dim
        assert self.physical_pos_embed_dim > 0, "模态嵌入维度不能大于等于总嵌入维度"
        print(f"物理位置编码维度: {self.physical_pos_embed_dim}, 模态嵌入维度: {modality_embed_dim}")
            

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr    
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    use_film=use_film_metadata,
                    condition_dim=condition_dim, 
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        if self.depth == 12:
            self.feature_indices = [3, 7, 11] # 对应第4, 8, 12层
        elif self.depth == 24:
            self.feature_indices = [7, 15, 23]
        # self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(len(self.feature_indices))])

        self.feature_fusion = PANetFusion3D(embed_dim=embed_dim)
        
        # YOLO检测头 - 使用完整的embed_dim输入通道
        self.yolo_head = singleLayerHead3D(num_anchors=6, num_class=6, channel=embed_dim//4)
        self.yolo_head1 = singleLayerHead3D(num_anchors=6, num_class=6, channel=embed_dim//2)
        self.yolo_head2 = singleLayerHead3D(num_anchors=6, num_class=6, channel=embed_dim)

        # 初始化权重
        self.patch_embed.apply(segm_init_weights)
        self.yolo_head.apply(segm_init_weights)
        self.yolo_head1.apply(segm_init_weights)
        self.yolo_head2.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)
    
    
    
    def forward_features(self, x, pos_embed, condition=None):
        """特征提取 - 返回多个中间层特征图
        
        Args:
            x: 输入数据 (B, C, H, W, D)
            batch_dataset_ids: 数据集标识符 (B,)
            batch_params: 数据集参数列表
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, C)
        x = x + pos_embed
        # x = self.pos_drop(x)

        B, N, C = x.shape

        features = []
        # feature_count = 0
        for i, layer in enumerate(self.layers):
            x = layer(x, cond_vector=condition)
            
            if i in self.feature_indices:
                features.append(x)
                # features.append(self.norm_layers[feature_count](x))
                # feature_count += 1
        
        return features  # 返回多个特征图
    
    def tokens_to_3d_feature(self, tokens):
        """将tokens转换回3D特征图"""
        B, N, C = tokens.shape
        H, W, D = self.patch_embed.grid_size
        x = tokens.transpose(1, 2).view(B, C, H, W, D)  # (B, C, H, W, D)
        
        return x
    
    def forward(self, x, condition, batch_params):

        if batch_params is not None:
            batch_params_tensor, has_velocity_mask = prepare_batch_params_tensor(batch_params, x.device)
            physical_coords = generate_physical_coords_fully_vectorized(
                batch_params_tensor, tuple(self.grid_tensor.tolist()), has_velocity_mask, x.device
            )
            # 生成物理位置编码
            physical_pos_embed = get_3d_physical_pos_embed_dynamic(self.physical_pos_embed_dim, physical_coords)
            
            # 生成模态类型嵌入
            modality_ids = has_velocity_mask.long()
            modality_embed = self.modality_embedding(modality_ids).unsqueeze(1)
            modality_embed = modality_embed.repeat(1, physical_pos_embed.shape[1], 1)
            
            # 合并物理位置编码和模态嵌入
            pos_embed = torch.cat([physical_pos_embed, modality_embed], dim=-1)
        else:
            print("警告: 没有提供物理参数，回退到标准的3D正弦位置编码。")
            pos_embed_base = get_3d_sincos_pos_embed(
                self.embed_dim, tuple(self.grid_tensor.tolist()), cls_token=False
            )
            pos_embed = torch.from_numpy(pos_embed_base).float().unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)



        multi_features = self.forward_features(x, pos_embed, condition=condition)
        
        # 将多个token特征转换为3D特征图
        grid_features = [self.tokens_to_3d_feature(features) for features in multi_features]
        
        # 使用PANet进行特征融合
        feature0, feature1, feature2 = self.feature_fusion(grid_features)
        # print(feature0.shape)
        # print(feature1.shape)
        # print(feature2.shape)
        # YOLO检测头
        out0 = self.yolo_head(feature0)
        out1 = self.yolo_head1(feature1)
        out2 = self.yolo_head2(feature2)
        
        return out1, out2, out0  # 与radtr_yolo.py保持一致的返回顺序

# ========= 模型注册 =========
@register_model
def raddet_tiny(pretrained=False, **kwargs):
    model = RadDet(embed_dim=192, depth=12,  **kwargs)
    return model

# @register_model
# def raddet_tiny_bimamba_none(pretrained=False, **kwargs):
#     model = RadDet(embed_dim=192, depth=12, if_bimamba=False, bimamba_type="none", use_pgm=False, **kwargs)
#     return model

# @register_model
# def raddet_tiny_bimamba_none(pretrained=False, **kwargs):
#     model = RadDet(embed_dim=192, depth=12, if_bimamba=False, bimamba_type="none", **kwargs)
#     return model

# @register_model
# def raddet_tiny_bimamba_v4(pretrained=False, **kwargs):
#     model = RadDet(embed_dim=192, depth=12, if_bimamba=False, bimamba_type="v4", **kwargs)
#     return model

@register_model
def raddet_tiny_NOphysical(pretrained=False, **kwargs):
    model = RadDet(embed_dim=192, depth=12, **kwargs)
    return model

@register_model
def raddet_base(pretrained=False, **kwargs):
    model = RadDet(embed_dim=384, depth=16, **kwargs)
    return model

@register_model
def raddet_large(pretrained=False, **kwargs):
    model = RadDet(embed_dim=768, depth=24, **kwargs)
    return model


if __name__ == '__main__':
    # ===================================================================
    #                           测试设置
    # ===================================================================
    print("测试 RadDet Tiny (Bi-Mamba None) 版本...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模拟配置
    config_data = {}
    config_model = {"input_shape": [256, 256, 64]}
    anchor_boxes = [[16, 16, 32], [32, 32, 64], [64, 64, 128]]
    
    # [修改] 定义批大小和条件维度
    BATCH_SIZE = 2
    CONDITION_DIM = 12 # 与模型定义的默认值 condition_dim=12 匹配
    
    print("\n" + "="*80)
    print(f"📋 开始测试 raddet_tiny_bimamba_none (Batch Size={BATCH_SIZE}, 带FiLM条件注入)")
    print("="*80)
    
    # ===================================================================
    #                         创建模拟输入
    # ===================================================================
    print("🔌 1. 准备模拟输入...")
    
    # a) 主输入张量 (RAD data)
    x_input = torch.randn(BATCH_SIZE, 1, 256, 256, 64).to(device)
    print(f"   - 输入张量 (x) shape: {x_input.shape}")
    
    # b) [新增] FiLM 条件向量
    #    这个向量模拟了从元数据编码器输出的结果
    mock_condition_vector = torch.rand(BATCH_SIZE, CONDITION_DIM).to(device)
    print(f"   - FiLM条件向量 (condition) shape: {mock_condition_vector.shape}")
    
    # c) PIPE 物理参数
    #    模拟一个包含不同类型数据的批次 (一个有速度，一个没有)
    mock_batch_params = [
        {'range_resolution': torch.tensor(0.1953), 'angular_resolution': torch.tensor(0.418), 'velocity_resolution': torch.tensor(0.4196), 'has_velocity': torch.tensor(True)},
        {'range_resolution': torch.tensor(0.115), 'angular_resolution': torch.tensor(0.469), 'velocity_resolution': torch.tensor(0.033), 'has_velocity': torch.tensor(False)} # 模拟CRUW数据
    ]
    # 手动将字典中的张量堆叠起来，以模拟DataLoader的collate_fn行为
    collated_batch_params = {k: torch.stack([d[k] for d in mock_batch_params]) for k in mock_batch_params[0]}
    print(f"   - 物理参数 (batch_params) 数量: {len(mock_batch_params)}")
    
    # ===================================================================
    #                         模型创建与评估
    # ===================================================================
    try:
        # 1. 创建模型
        print("\n🛠️ 2. 创建模型实例...")
        model = raddet_tiny(
            config_data=config_data,
            config_model=config_model,
            anchor_boxes=anchor_boxes,
            condition_dim=CONDITION_DIM # 显式传递以确保匹配
        ).to(device).eval()
        
        print(f"✅ 实际配置: embed_dim={model.embed_dim}, depth={model.depth}, condition_dim={CONDITION_DIM}")
        
        # 2. 模型大小评估
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 模型参数数量: {total_params/1e6:.2f}M")
        
        # 3. FLOPs 分析
        print("\n🔬 3. 开始进行详细的 FLOPs 分析 (使用单个样本)...")
        # 为FLOPs分析准备单一样本输入
        single_input_for_flops = x_input[0:1]
        single_condition_for_flops = mock_condition_vector[0:1]
        single_params_for_flops = {k: v[0:1] for k, v in collated_batch_params.items()}

        try:
            # [修改] 在FlopCountAnalysis中传入所有参数
            flop_counter = FlopCountAnalysis(model, (single_input_for_flops, single_condition_for_flops, single_params_for_flops))
            total_flops = flop_counter.total()
            print(f"📈 总 GFLOPs: {total_flops / 1e9:.4f} G")
            
            # (可选) 打印详细的模块分析报告...
            
        except Exception as flops_error:
            print(f"⚠️ FLOPs 计算过程中发生错误: {flops_error}")
            import traceback
            traceback.print_exc()

        # 4. 推理和计时
        print("\n⏱️ 4. 开始进行推理时间测试...")
        
        with torch.no_grad():
            if device.type == 'cuda':
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                # 预热
                for _ in range(5):
                    _ = model(x_input, condition=mock_condition_vector, batch_params=collated_batch_params)
                
                start_time.record()
                # [修改] 传入所有参数进行推理
                outputs = model(x_input, condition=mock_condition_vector, batch_params=collated_batch_params)
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
            else:
                import time
                start = time.time()
                outputs = model(x_input, condition=mock_condition_vector, batch_params=collated_batch_params)
                end = time.time()
                inference_time = (end - start) * 1000

        print(f"📤 输出1形状: {outputs[0].shape} (设备: {outputs[0].device})")
        print(f"📤 输出2形状: {outputs[1].shape} (设备: {outputs[1].device})")
        print(f"📤 输出3形状: {outputs[2].shape} (设备: {outputs[2].device})")
        print(f"⏱️ 推理时间 ({BATCH_SIZE}个样本): {inference_time:.2f}ms")
        print(f"\n✅ raddet_tiny_bimamba_none 版本测试成功！")
        
        del model, outputs, x_input
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ raddet_tiny_bimamba_none 版本测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("🎉 测试完成！")
    print("="*80)