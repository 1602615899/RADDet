# -*- coding: utf-8 -*-
"""
此脚本定义了用于处理 SCORP 数据集的 PyTorch Dataset 类。

这是用于训练和评估的最终版本，其中包含了：
- 预先计算好的统计数据。
- SCORP 特有的预处理（切片和插值）。
- 将数据 resize 到统一维度的功能。
- 用于验证和调试的随机可视化工具。

所需依赖: torch, numpy, tqdm, scikit-image, matplotlib
请确保已安装:
pip install scikit-image matplotlib
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import json
import random
from scipy.ndimage import zoom
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# 模块级辅助函数 (数据处理)
# ==============================================================================
def normalize_data(data):
    """
    Normalize numpy array to the range [0, 1].
    """
    data_min = data.min()
    data_max = data.max()
    
    # Add a small epsilon to avoid division by zero
    if (data_max - data_min) < 1e-8:
        return np.zeros_like(data)
        
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

def readRAD(filename):
    """安全地读取一个 .npy 文件。"""
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None

def getLog(target_array, scalar=1., log_10=True):
    """计算数组的对数值。"""
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return scalar * np.log(target_array + 1.)

# ==============================================================================
# 模块级辅助函数 (可视化)
# ==============================================================================

def plot_RAD(rad_data, params, root_dir, i, RA=True, RD=True, AD=True):
    """
    主绘图函数，生成并保存RAD视图。
    假设 rad_data 的维度顺序是 (Range, Angle, Doppler)。
    """
    os.makedirs(root_dir, exist_ok=True)
    data = rad_data.squeeze().cpu().numpy()
    
    RA_data = np.mean(data, axis=2) # -> (Range, Angle)
    RD_data = np.mean(data, axis=1) # -> (Range, Doppler)
    AD_data = np.mean(data, axis=0) # -> (Angle, Doppler)

    if RA:
        plot_RA(RA_data, params, root_dir, i)
    if RD:
        plot_RD(RD_data, params, root_dir, i)
    if AD:
        plot_AD(AD_data, params, root_dir, i)

def plot_RA(RA_data, params, root_dir, i):
    """绘制并保存Range-Angle视图。"""
    max_range = params.get('max_range', 1.0)
    fov = params.get('azimuth_fov', 90.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(RA_data, aspect='auto', origin='lower', extent=[-fov/2, fov/2, 0, max_range])
    plt.colorbar(label='Amplitude')
    plt.title(f'Range-Angle View - {i}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Range (m)')
    plt.savefig(os.path.join(root_dir, f'RA_{i}.png'))
    plt.close()

def plot_RD(RD_data, params, root_dir, i):
    """绘制并保存Range-Doppler视图。"""
    max_range = params.get('max_range', 1.0)
    max_vel = params.get('max_velocity', 1.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(RD_data, aspect='auto', origin='lower', extent=[-max_vel, max_vel, 0, max_range])
    plt.colorbar(label='Amplitude')
    plt.title(f'Range-Doppler View - {i}')
    plt.xlabel('Doppler (m/s)')
    plt.ylabel('Range (m)')
    plt.savefig(os.path.join(root_dir, f'RD_{i}.png'))
    plt.close()

def plot_AD(AD_data, params, root_dir, i):
    """绘制并保存Angle-Doppler视图。"""
    fov = params.get('azimuth_fov', 90.0)
    max_vel = params.get('max_velocity', 1.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(AD_data, aspect='auto', origin='lower', extent=[-max_vel, max_vel, -fov/2, fov/2])
    plt.colorbar(label='Amplitude')
    plt.title(f'Angle-Doppler View - {i}')
    plt.xlabel('Doppler (m/s)')
    plt.ylabel('Angle (degrees)')
    plt.savefig(os.path.join(root_dir, f'AD_{i}.png'))
    plt.close()

# ==============================================================================
# SCORP 数据集类
# ==============================================================================


NORMALIZATION_STATS = {
    'azimuth_fov': {'min': 90.0, 'max': 180.0},
    'max_range': {'min': 14.24, 'max': 118.0},
    'max_velocity': {'min': 0.0, 'max': 23.02},
    'range_resolution': {'min': 0.047, 'max': 0.97},
    'velocity_resolution': {'min': 0.0, 'max': 0.48},
    'angular_resolution': {'min': 0.352, 'max': 1.0}
}
NUMERICAL_KEYS_FOR_FILM = list(NORMALIZATION_STATS.keys())

def normalize_condition_vector_for_film(vector, ordered_keys):
    """为FiLM的条件向量进行标准的 [0, 1] Min-Max 归一化。"""
    vector = np.array(vector, dtype=np.float32)
    for i, key in enumerate(ordered_keys):
        if key in NUMERICAL_KEYS_FOR_FILM:
            stats = NORMALIZATION_STATS[key]
            min_val, max_val = stats['min'], stats['max']
            if max_val > min_val:
                vector[i] = (vector[i] - min_val) / (max_val - min_val)
            else:
                vector[i] = 0.0
    return vector


class SCORP(Dataset):
    def __init__(self, data_root, sensor_params, sequences=None):
        self.data_root = data_root
        self.dataset_params = sensor_params
        
        # 硬编码预先计算好的统计值
        self.mean_log = 1.792864
        self.std_log = 0.388794

        # 统一目标尺寸 (Range, Angle, Doppler)
        self.target_shape = (256, 256, 64)
        
        if sequences is None:
            self.sequences = glob.glob(os.path.join(data_root, "2019-*-*-*-*-*"))
        else:
            self.sequences = sequences
            
        self.all_npy_files = []
        self._collect_all_npy_files()
        self.raw_params = sensor_params
        self.ordered_keys = [
            'sensor_awr', 'sensor_retina', 'platform_mobile', 'platform_static', 'platform_mixed',
            'azimuth_fov', 'max_range', 'max_velocity', 'range_resolution',
            'velocity_resolution', 'angular_resolution', 'has_velocity'
        ]
        
        # 从原始参数创建向量
        condition_vector_list  = [self.raw_params[key] for key in self.ordered_keys]
        
        # 应用标准 [0, 1] 归一化，得到用于FiLM的向量
        normalized_vector = normalize_condition_vector_for_film(condition_vector_list, self.ordered_keys)

        # 存储最终的FiLM条件张量
        self.condition_tensor = torch.tensor(normalized_vector, dtype=torch.float32)
        
        

    def _collect_all_npy_files(self):
        """预先收集所有.npy文件路径以提高效率。"""
        print("正在收集所有SCORP数据文件路径...")
        for sequence_path in tqdm(self.sequences, desc="扫描序列目录"):
            rad_numpy_paths = glob.glob(os.path.join(sequence_path, "ral_outputs_*", "RAD_numpy"))
            for rad_numpy_path in rad_numpy_paths:
                npy_files = sorted(glob.glob(os.path.join(rad_numpy_path, "*.npy")))
                self.all_npy_files.extend(npy_files)
        print(f"总共找到 {len(self.all_npy_files)} 个SCORP数据文件")

    def __len__(self):
        return len(self.all_npy_files)

    def __getitem__(self, idx):
        RAD_filename = self.all_npy_files[idx]
        RAD_data = readRAD(RAD_filename)
        if RAD_data is None:
            raise ValueError(f"RAD文件读取失败: {RAD_filename}")
        
        # SCORP 特有预处理 1: 切片，保留远距离部分
        if RAD_data.shape[0] == 256:
            RAD_data = RAD_data[128:, :, :] # (128, 256, 64)
        
        # SCORP 特有预处理 2: 插值，将距离维从128扩展到256
        if RAD_data.shape == (128, 256, 64):
            zoom_factors = (2.0, 1.0, 1.0) # 只在第一个维度（Range）上插值
            RAD_data = zoom(RAD_data, zoom_factors, order=1)
        
        # 确保形状符合预期
        if RAD_data.shape != (256, 256, 64):
            # 如果插值后尺寸不匹配（例如原始尺寸不对），则进行强制resize
            RAD_data = resize(RAD_data, (256, 256, 64), anti_aliasing=True)
        
        # 标准数据处理流程
        # 假设SCORP的.npy文件已经是幅度数据，直接进行对数变换
        RAD_data = getLog(RAD_data, log_10=True)
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        RAD_data = normalize_data(RAD_data)
        
        RAD_data = RAD_data[np.newaxis, :, :, :]
        
        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'condition': self.condition_tensor, # The normalized tensor for the model
            'raw_params': self.raw_params       # The original dict for visualization & pos encoding
        }

# ==============================================================================
# SCORP 数据集工厂函数
# ==============================================================================

def Create_SCORP_Pretrain_Dataset(data_root, sensor_config_path):
    """ 创建SCORP预训练数据集 """
    print("-" * 60)
    print("开始加载 SCORP 数据集...")

    config_key = "SCORP"

    if not (sensor_config_path and os.path.exists(sensor_config_path)):
        raise FileNotFoundError(f"传感器配置文件未找到: {sensor_config_path}")
    
    with open(sensor_config_path, 'r') as f:
        sensor_configs = json.load(f)

    if config_key not in sensor_configs:
        raise KeyError(f"在配置文件中未找到配置键: '{config_key}'")
    
    sensor_params = sensor_configs[config_key]

    print(f"已成功加载 '{config_key}' 的传感器配置。")

    return SCORP(data_root, sensor_params=sensor_params)

# ==============================================================================
# 主执行块: 用于验证和随机可视化
# ==============================================================================
if __name__ == '__main__':
    # --- 用户配置区域 ---
    SCORP_DATA_ROOT = "/media/ljm/Raid/ChenHongliang/20190813_scorp_dataset"
    SENSOR_CONFIG_FILE = "/media/ljm/Raid/ChenHongliang/RAGM/datasets/sensor_params.json"
    PLOT_OUTPUT_DIR = "./plot_outputs"
    NUM_PLOTS = 5
    # --- 配置结束 ---

    try:
        # 步骤 1: 实例化 SCORP 数据集
        print("="*60)
        dataset = Create_SCORP_Pretrain_Dataset(
            data_root=SCORP_DATA_ROOT,
            sensor_config_path=SENSOR_CONFIG_FILE
        )
        dataset_name = dataset.dataset_params.get('dataset_name', 'Unknown')
        print(f"成功实例化 '{dataset_name}' 数据集，大小: {len(dataset)} 个样本")
        print(f"使用的均值: {dataset.mean_log:.6f}, 标准差: {dataset.std_log:.6f}")
        print("="*60)
        
        # 步骤 2: 随机选择样本进行可视化
        if len(dataset) > 0:
            print(f"\n随机选择 {NUM_PLOTS} 个样本进行处理和可视化...")
            print(f"绘图将保存到: {os.path.abspath(PLOT_OUTPUT_DIR)}")
            
            random_indices = random.sample(range(len(dataset)), k=min(NUM_PLOTS, len(dataset)))
            
            for idx in random_indices:
                print(f"\n--- 正在处理样本索引: {idx} ---")
                sample = dataset[idx]
                data_tensor = sample['data']
                file_identifier = f"{dataset_name}_sample_{idx}"
                
                print(f"  最终数据张量形状 (C, R, A, D): {data_tensor.shape}")
                print(f"  数据范围: [{data_tensor.min():.4f}, {data_tensor.max():.4f}]")
                
                plot_RAD(
                    rad_data=data_tensor,
                    params=dataset.dataset_params,
                    root_dir=PLOT_OUTPUT_DIR,
                    i=file_identifier
                )
                print(f"  已为样本 {idx} 生成并保存视图。")
        else:
            print("\n数据集为空，无法进行可视化。")

    except Exception as e:
        print(f"\n" + "!"*60)
        print(f"!!! 脚本执行过程中发生严重错误: {e}")
        print("!"*60)
        import traceback
        traceback.print_exc()