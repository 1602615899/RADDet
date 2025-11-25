import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, Tuple, Union
import torch.nn.functional as F

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import to_2tuple, DropPath
from timm.models.vision_transformer import _load_weights
from einops import rearrange, repeat
import math
# from utils.pos_embed import * ### 修改：我将直接定义3D位置编码函数
from collections import namedtuple

# 假设 mamba_simple 和 mamba_ssm 在python路径中
from mamba_ssm.modules.mamba_simple import Mamba 


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("警告：未找到RMSNorm，回退到nn.LayerNorm。")


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
        # 为了训练稳定性，通常初始化时让gamma接近1，beta接近0
        # 这里通过加1来实现，使得在网络初始化时，FiLM层近似于一个恒等变换
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

# ==================== [新函数：实现PIPE理念的归一化] ====================
def normalize_physical_coords_pipe(coords: Tensor, has_velocity_mask: Tensor) -> Tensor:
    """
    对物理坐标进行对称性保持归一化，严格遵循PIPE框架理念。
    - 距离/时间 (第0和2列，如果非速度模式): 归一化到 [0, 1]
    - 角度/速度 (第1和2列，如果是速度模式): 归一化到 [-1, 1]

    Args:
        coords (Tensor): 形状为 [B, N, 3] 的原始物理坐标张量。
                         列顺序为 (距离, 角度, 速度/时间)。

    Returns:
        Tensor: 归一化后的坐标张量。
    """
    B, N, C = coords.shape
    normalized_coords = torch.zeros_like(coords)

    # 1. 处理距离 (第0列) - 非对称，归一化到 [0, 1]
    dist_max = torch.max(coords[:, :, 0], dim=1, keepdim=True)[0]
    # 防止除以零
    dist_max[dist_max == 0] = 1.0
    normalized_coords[:, :, 0] = coords[:, :, 0] / dist_max

    # 2. 处理角度 (第1列) - 对称，归一化到 [-1, 1]
    angle_abs_max = torch.max(torch.abs(coords[:, :, 1]), dim=1, keepdim=True)[0]
    angle_abs_max[angle_abs_max == 0] = 1.0
    normalized_coords[:, :, 1] = coords[:, :, 1] / angle_abs_max

    # 3. 速度/时间 (根据 has_velocity_mask 动态选择)
    vel_time_coords = coords[:, :, 2]
    
    # 找出每个样本的最大绝对值，用于[-1, 1]归一化
    abs_max = torch.max(torch.abs(vel_time_coords), dim=1, keepdim=True)[0]
    abs_max[abs_max == 0] = 1.0
    symmetric_norm = vel_time_coords / abs_max # 归一化到 [-1, 1]
    
    # 找出每个样本的最大值，用于[0, 1]归一化
    max_val = torch.max(vel_time_coords, dim=1, keepdim=True)[0]
    max_val[max_val == 0] = 1.0
    asymmetric_norm = vel_time_coords / max_val # 归一化到 [0, 1]

    # 使用 has_velocity_mask 作为条件选择器
    # has_velocity_mask 需要扩展维度以匹配坐标张量
    mask = has_velocity_mask.view(B, 1).expand(B, N)
    normalized_coords[:, :, 2] = torch.where(mask, symmetric_norm, asymmetric_norm)
    
    return normalized_coords

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
    [已修改] 完全向量化的物理坐标生成，并使用PIPE归一化。
    """
    B = batch_params_tensor.shape[0]
    H, W, D = grid_size
    
    # ... (步骤 1-3：索引生成和原始坐标计算，保持不变) ...
    h_indices = torch.arange(H, device=device).view(1, H, 1, 1).expand(B, H, W, D)
    w_indices = torch.arange(W, device=device).view(1, 1, W, 1).expand(B, H, W, D)
    d_indices = torch.arange(D, device=device).view(1, 1, 1, D).expand(B, H, W, D)
    
    range_res = batch_params_tensor[:, 0].view(B, 1, 1, 1)
    angle_res = batch_params_tensor[:, 1].view(B, 1, 1, 1)
    vel_or_time_res = batch_params_tensor[:, 2].view(B, 1, 1, 1)
    
    distances = h_indices * range_res
    angles = (w_indices - W / 2) * angle_res
    
    velocities_or_times = torch.where(
        has_velocity_mask.view(B, 1, 1, 1),
        (d_indices - D / 2) * vel_or_time_res,
        d_indices * vel_or_time_res
    )
    
    # 步骤 4: 堆叠并展平 (保持不变)
    coords = torch.stack([distances, angles, velocities_or_times], dim=-1).view(B, -1, 3)
    
    # [关键修正] 步骤 5: 使用新的PIPE对称性保持归一化，替换掉旧的min-max归一化
    coords_normalized = normalize_physical_coords_pipe(coords, has_velocity_mask)
    
    return coords_normalized

def prepare_batch_params_tensor(batch_params_dict, device):
    """
    [最终确认版] 将 DataLoader collate 后的参数字典转换为张量格式。
    这个版本能够完美处理您对CRUW参数的新用法。
    """
    B = batch_params_dict['range_resolution'].shape[0]
    
    batch_params_tensor = torch.zeros(B, 4, device=device)
    has_velocity_mask = torch.zeros(B, dtype=torch.bool, device=device)
    
    # 直接从字典中获取整个批次的参数张量
    range_res_batch = batch_params_dict['range_resolution'].to(device, dtype=torch.float32)
    angle_res_batch = batch_params_dict['angular_resolution'].to(device, dtype=torch.float32)
    has_velocity_batch = batch_params_dict['has_velocity'].to(device, dtype=torch.bool)
    
    # 智能选择第三个分辨率
    vel_res_batch = batch_params_dict.get('velocity_resolution', torch.zeros(B, device=device)).to(device, dtype=torch.float32)
    
    # [关键逻辑]
    # 对于CRUW:
    #   has_velocity_batch 会是 False。
    #   vel_res_batch 会是从JSON加载的 tensor([1., ...])。
    #   torch.where 会选择 vel_res_batch 的值。
    # 对于其他数据集:
    #   has_velocity_batch 会是 True。
    #   torch.where 会选择 vel_res_batch 的值。
    # 
    # 所以，我们不再需要 time_resolution 这个备用方案了，逻辑被简化了！
    third_dim_res_batch = vel_res_batch 
    
    # 填充最终的张量
    batch_params_tensor[:, 0] = range_res_batch
    batch_params_tensor[:, 1] = torch.deg2rad(angle_res_batch)
    batch_params_tensor[:, 2] = third_dim_res_batch
    batch_params_tensor[:, 3] = has_velocity_batch.float()
    
    has_velocity_mask = has_velocity_batch
    
    return batch_params_tensor, has_velocity_mask

# --- 位置编码结束 ---


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask):
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # kv形状是(B, N, C)，与q相同。这意味着编码器的kv应该与解码器中的q具有相同维度
        kv_B, kv_N, kv_C = kv.shape
        kv = self.kv(kv).reshape(kv_B, kv_N, 2, self.num_heads, kv_C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
             attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn2 = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, mask):
        # 在DecoderBlock中，q是自回归token，kv是来自编码器的特征
        q = q + self.drop_path(self.attn2(self.norm2_1(q), self.norm2_2(kv), mask))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q



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


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight", "w3.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d): # 修改：处理Conv3d
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    """Masked Autoencoder with VisionMamba backbone"""
    def __init__(self,
                 input_size = (256, 256, 64),
                 patch_size = (8, 8, 8),
                 embed_dim=192,
                 dec_embed_dim=192,
                 depth=12,
                 ssm_cfg=None, 
                 drop_path_rate=0.1,
                 pos_drop_rate=0.0, # ### FIX 1: 添加 pos_drop_rate 参数
                 modality_embed_dim=24, # ### FIX 2: 添加 modality_embed_dim 参数
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
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
                 group_size=(2, 2, 2), # [关键修改] 新增参数，定义分组的大小 (在H,W,D维度上)
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        if depth == 12:
            self.skip = [6, 8, 10, 12]
        elif depth == 24:
            self.skip = [12, 16, 20, 24]
        else:
            num_skips = 4
            self.skip = [int(depth / num_skips * (i + 1)) for i in range(num_skips)]
        print(f"使用跳跃连接的层: {self.skip}")
        

        # 动态计算网格大小和分组大小
        self.grid_size = (
            input_size[0] // patch_size[0],
            input_size[1] // patch_size[1],
            input_size[2] // patch_size[2]
        )
        
        print(f"网格大小: {self.grid_size}")
        

        self.num_groups_h = self.grid_size[0] // self.group_size[0]
        self.num_groups_w = self.grid_size[1] // self.group_size[1]
        self.num_groups_d = self.grid_size[2] // self.group_size[2]
        self.num_groups = self.num_groups_h * self.num_groups_w * self.num_groups_d
        self.tokens_per_group = self.group_size[0] * self.group_size[1] * self.group_size[2]

        assert self.grid_size[0] % self.group_size[0] == 0, "Grid height must be divisible by group height"
        assert self.grid_size[1] % self.group_size[1] == 0, "Grid width must be divisible by group width"
        assert self.grid_size[2] % self.group_size[2] == 0, "Grid depth must be divisible by group depth"

        print(f"分组数量: {self.num_groups} ({self.num_groups_h}x{self.num_groups_w}x{self.num_groups_d})")
        print(f"每组Token数量: {self.tokens_per_group}")

        self.register_buffer("grid_tensor", torch.tensor(self.grid_size, dtype=torch.long))

        # 模态类型嵌入 (保持分辨率信息)
        self.modality_embedding = nn.Embedding(2, modality_embed_dim) # ### FIX 2: 使用参数
        
        # 调整物理位置编码维度
        self.physical_pos_embed_dim = embed_dim - modality_embed_dim
        assert self.physical_pos_embed_dim > 0, "模态嵌入维度不能大于等于总嵌入维度"
        print(f"物理位置编码维度: {self.physical_pos_embed_dim}, 模态嵌入维度: {modality_embed_dim}")
        
        # 编码器 (保持分辨率信息)
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(
            input_size=input_size,
            patch_size=patch_size, 
            in_chans=1,  # RADDet使用1个通道
            embed_dim=embed_dim
        )
        
        self.pos_drop = nn.Dropout(p=pos_drop_rate) # ### FIX 1: 初始化 pos_drop
        
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
        
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(len(self.skip))])

        # 解码器 (使用交叉注意力)
        self.dec_embed_dim = dec_embed_dim
        self.enc2dec = nn.Linear(embed_dim * len(self.skip), self.dec_embed_dim * len(self.skip))

        self.ar_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        
        # 使用你定义的 DecoderBlock (交叉注意力)
        self.dec_block = nn.ModuleList([
            DecoderBlock(self.dec_embed_dim, self.dec_embed_dim // 64, 4, qkv_bias=True, qk_scale=None,
                         norm_layer=nn.LayerNorm) for _ in range(len(self.skip))
        ])
        self.ar_norm = nn.LayerNorm(self.dec_embed_dim)

        ### 修改：预测头输出维度必须匹配展平的patch大小
        patch_dim = np.prod(patch_size) * 1
        self.ar_pred = nn.Linear(self.dec_embed_dim, patch_dim)
        print(f"预测头输出维度: {patch_dim}")

        # 初始化
        self.patch_embed.apply(segm_init_weights)
        trunc_normal_(self.ar_token, std=.02)
        self.apply(partial(_init_weights, n_layer=depth, n_residuals_per_layer=2))
        self.dec_block.apply(self.atten_init_weights)

        num_groups_to_predict = self.num_groups - 1
        self.register_buffer("causal_mask", self.mask_generate(num_groups_to_predict, self.tokens_per_group))

    @staticmethod
    def mask_generate(num_groups, tokens_per_group):
        # 为交叉注意力创建因果mask
        mask = torch.tril(torch.ones((num_groups, num_groups), dtype=torch.float))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        # 放大掩码以匹配每个组内的token数量
        mask = torch.repeat_interleave(mask, repeats=tokens_per_group, dim=0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_group, dim=1)
        return mask

    def atten_init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"ar_token"}

    def forward_features(self, x, pos_embed, condition=None):
        B, C, H, W, D = x.shape

        x = self.patch_embed(x)
        B, N, C_embed = x.shape

        x = x + pos_embed
        x = self.pos_drop(x)
        
        gh, gw, gd = self.grid_size
        x = x.reshape(B, gh, gw, gd, self.embed_dim)

        # 2. 使用einops进行分组
        ph, pw, pd = self.group_size
        x_grouped = rearrange(x, 'b (h ph) (w pw) (d pd) c -> b (h w d) (ph pw pd) c', 
                              ph=ph, pw=pw, pd=pd)
        
        # [关键修改] 规避信息泄露：编码器只看到除最后一组外的所有组
        # x_grouped 形状: [B, Num_Groups, Tokens_per_Group, C]
        # 我们将组和组内token维度合并，然后丢弃最后一组
        hidden_states = x_grouped[:, :-1, :, :].reshape(B, -1, self.embed_dim)
        
        features = []
        count = 0
        skip_idx = 0
        for layer in self.layers:
            hidden_states = layer(hidden_states, cond_vector=condition)
            count += 1
            if count in self.skip:
                features.append(self.norm_layers[skip_idx](hidden_states))
                skip_idx += 1
                
        # 连接来自不同深度的特征
        features = torch.cat(features, dim=-1)
        
        # 投影到解码器维度
        features = self.enc2dec(features)
        
        B, Num_Tokens, Dec_C_total = features.shape
        num_skips = len(self.skip)
        # 重塑为(B, Num_Tokens, Dec_Dim, Num_Skips)以供解码器块使用
        features = features.reshape(B, Num_Tokens, self.dec_embed_dim, num_skips)

        return features

    def forward_decoder(self, latent_ar, decoder_pos_embed):
        B_lat, N_lat, C_dec, D_dec = latent_ar.shape

        # 创建解码器输入：AR token + 所有patch的位置编码
        ar_token = self.ar_token + decoder_pos_embed
        
        # [关键修改] 同样对ar_token(Query)进行分组
        gh, gw, gd = self.grid_size
        ar_token_grid = ar_token.reshape(ar_token.shape[0], gh, gw, gd, C_dec)
        ph, pw, pd = self.group_size
        ar_token_grouped = rearrange(ar_token_grid, 'b (h ph) (w pw) (d pd) c -> b (h w d) (ph pw pd) c', 
                                     ph=ph, pw=pw, pd=pd)
        
        # [关键修改] 解码器也只处理 n-1 组，以匹配损失函数和掩码
        ar_token_to_predict = ar_token_grouped[:, :-1, :, :].reshape(ar_token.shape[0], -1, C_dec)

        if ar_token_to_predict.shape[0] != B_lat:
            ar_token_to_predict = ar_token_to_predict.expand(B_lat, -1, -1)
        
        # 应用解码器块
        for i, blk in enumerate(self.dec_block):
            kv_features = latent_ar[:, :, :, i]  # [B_lat, N_lat, C_dec]
            ar_token_to_predict = blk(ar_token_to_predict, kv_features, self.causal_mask)

        ar_token_to_predict = self.ar_norm(ar_token_to_predict)
        ar_token_to_predict = self.ar_pred(ar_token_to_predict)
        return ar_token_to_predict

    ### 修改：完全3D patchify函数
    def patchify(self, imgs):
        """
        imgs: (B, C_in, H, W, D)
        x: (B, L, patch_size_h*patch_size_w*patch_size_d*C_in)
        """
        p_h, p_w, p_d = self.patch_size
        assert imgs.shape[2] % p_h == 0 and imgs.shape[3] % p_w == 0 and imgs.shape[4] % p_d == 0
        
        x = rearrange(imgs, 'b c (h ph) (w pw) (d pd) -> b (h w d) (ph pw pd c)', 
                      ph=p_h, pw=p_w, pd=p_d)
        return x

    ### 修改：带正确3D重塑的损失计算
    def forward_loss(self, imgs, pred):
        target = self.patchify(imgs)

        # Optional: per-patch normalization
        if False:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        B, N_total, C_patch = target.shape
        gh, gw, gd = self.grid_size
        target_grid = target.reshape(B, gh, gw, gd, C_patch)

        ph, pw, pd = self.group_size
        target_grouped = rearrange(target_grid, 'b (h ph) (w pw) (d pd) c -> b (h w d) (ph pw pd) c',
                                   ph=ph, pw=pw, pd=pd)
        
        # 目标是 n-1 组
        target_to_predict = target_grouped[:, :-1, :, :].reshape(B, -1, C_patch)

        assert pred.shape == target_to_predict.shape, f"形状不匹配: pred {pred.shape} vs target {target_to_predict.shape}"
        
        loss = (pred - target_to_predict) ** 2
        return loss

    def forward(self, x, condition, batch_params):
        # 输入x也是重建的标签
        labels = x.clone()
        
        num_patches = self.patch_embed.num_patches
        seq_len_for_mask = num_patches - 1  # 编码器看到 n-1 组

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
        
        # 解码器使用标准网格位置编码 (不使用物理参数)
        dec_pos_embed_base = get_3d_sincos_pos_embed(self.dec_embed_dim, tuple(self.grid_tensor.tolist()), cls_token=False)
        decoder_pos_embed = torch.from_numpy(dec_pos_embed_base).float().unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)
        
        # 获取编码器特征
        # 1. 编码器只看到 n-1 组
        latent_features = self.forward_features(x, pos_embed, condition=condition)
        # 2. 解码器预测这 n-1 组
        predictions = self.forward_decoder(latent_features, decoder_pos_embed)
        # 3. 损失函数也只计算这 n-1 组
        loss = self.forward_loss(labels, predictions)

        return loss.mean()

@register_model
def vision_mamba_3d_tiny(pretrained=False, **kwargs):
    """ VisionMamba-3D Tiny用于预训练 """
    model = VisionMamba(
        patch_size=(8, 8, 8), 
        input_size=(256, 256, 64),
        embed_dim=192,     
        depth=12,           
        dec_embed_dim=96, 
        bimamba_type = "none",
        modality_embed_dim=24, 
        group_size=(4, 4, 1),
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def vision_mamba_3d_base(pretrained=False, **kwargs):
    """ VisionMamba-3D Base用于预训练 """
    model = VisionMamba(
        patch_size=(8, 8, 8), 
        input_size=(256, 256, 64),
        embed_dim=384,      
        depth=12,           
        dec_embed_dim=192, 
        modality_embed_dim=24,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def vision_mamba_3d_large(pretrained=False, **kwargs):
    """ VisionMamba-3D Large用于预训练 """
    model = VisionMamba(
        patch_size=(8, 8, 8), 
        input_size=(256, 256, 64),
        embed_dim=768,      
        depth=24,           
        dec_embed_dim=384, 
        modality_embed_dim=24,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_factories = { "Tiny": vision_mamba_3d_tiny }

    x = torch.randn(2, 1, 256, 256, 64).to(device) # 批大小为2
    print(f"\n创建测试输入张量，形状: {x.shape}")

    # [新增] 为测试创建符合新签名的虚拟条件输入
    # 1. 用于PIPE的虚拟原始参数字典列表
    #    注意：这里需要包含'range_res', 'angle_res'等PIPE函数内部使用的键
    dummy_raw_params_kradar = {
        'range_resolution': 0.46, 'angular_resolution': 0.418, 'velocity_resolution': 0.06, 'has_velocity': True,
        # 为 prepare_batch_params_tensor 提供别名
        'range_res': 0.46, 'angle_res': np.deg2rad(0.418), 'vel_res': 0.06
    }
    dummy_raw_params_cruw = {
        'range_resolution': 0.115, 'angular_resolution': 0.469, 'has_velocity': False,
        'range_res': 0.115, 'angle_res': np.deg2rad(0.469), 'time_res': 0.033
    }
    batch_params_for_pipe = [dummy_raw_params_kradar, dummy_raw_params_cruw]

    # 2. 用于FiLM的虚拟归一化条件张量
    condition_for_film = torch.rand(2, 12).to(device) # 形状: [batch_size, condition_dim]

    for name, factory in model_factories.items():
        print(f"\n{'='*20} 测试 VisionMamba-3D {name} {'='*20}")
        try:
            model = factory().to(device)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"模型实例化成功，参数量: {num_params / 1e6:.2f}M")

            # [修正] 使用正确的参数调用模型进行前向传播
            with torch.no_grad():
                output_loss = model(x, condition=condition_for_film, batch_params=batch_params_for_pipe)
            print(f"前向传播成功，输出损失: {output_loss.item():.4f}")
            print(f"{'='*20} {name} 模型测试通过 {'='*20}\n")
        except Exception as e:
            print(f"!!! VisionMamba-3D {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()