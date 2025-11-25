# -*- coding: utf-8 -*-
"""
此脚本定义了用于处理 K-Radar 数据集的 PyTorch Dataset 类。

这是用于训练和评估的最终版本，其中包含了：
- 预先计算好的统计数据。
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
from skimage.transform import resize
import matplotlib.pyplot as plt

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

def getMagnitude(target_array, power_order=2):
    """从复数数据中计算幅度（或功率）。"""
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array

def getLog(target_array, scalar=1., log_10=True):
    """计算数组的对数值。"""
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return scalar * np.log(target_array + 1.)

def preprocess_data(target_array):
    """对原始RAD数据进行标准的预处理流程: 复数 -> 功率 -> 对数尺度。"""
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

# ==============================================================================
# 模块级辅助函数 (可视化)
# ==============================================================================

def plot_RAD(rad_data, params, root_dir, i, RA=True, RD=True, AD=True):
    """
    主绘图函数，生成并保存RAD视图。
    假设 rad_data 的维度顺序是 (Doppler, Angle, Range)。
    """
    os.makedirs(root_dir, exist_ok=True)
    data = rad_data.squeeze().cpu().numpy()
    
    # 通过对特定轴求平均值来生成2D视图
    RA_data = np.mean(data, axis=0) # 对Doppler轴(0)求平均 -> (Angle, Range)
    RD_data = np.mean(data, axis=1) # 对Angle轴(1)求平均 -> (Doppler, Range)
    AD_data = np.mean(data, axis=2) # 对Range轴(2)求平均 -> (Doppler, Angle)

    if RA:
        plot_RA(RA_data, params, root_dir, i)
    if RD:
        plot_RD(RD_data, params, root_dir, i)
    if AD:
        # AD视图通常保持原样，Doppler在Y轴，Angle在X轴
        plot_AD(AD_data, params, root_dir, i)

def plot_RA(RA_data, params, root_dir, i):
    """
    绘制并保存Range-Angle视图。
    [已修改] 将 Range 放在 Y 轴。
    RA_data 原始形状: (Angle, Range)
    """
    max_range = params.get('max_range', 1.0)
    fov = params.get('azimuth_fov', 90.0)
    plt.figure(figsize=(8, 6))
    
    # 1. 转置数据: RA_data.T -> 形状变为 (Range, Angle)
    # 2. 交换 extent: [x_min, x_max, y_min, y_max] -> [angle_min, angle_max, range_min, range_max]
    # 3. 交换坐标轴标签
    plt.imshow(RA_data.T, aspect='auto', origin='lower', extent=[-fov/2, fov/2, 0, max_range])
    
    plt.colorbar(label='Amplitude')
    plt.title(f'Range-Angle View - {i}')
    plt.xlabel('Angle (degrees)') # X 轴是角度
    plt.ylabel('Range (m)')       # Y 轴是距离
    plt.savefig(os.path.join(root_dir, f'RA_{i}.png'))
    plt.close()

def plot_RD(RD_data, params, root_dir, i):
    """
    绘制并保存Range-Doppler视图。
    [已修改] 将 Range 放在 Y 轴。
    RD_data 原始形状: (Doppler, Range)
    """
    max_range = params.get('max_range', 1.0)
    max_vel = params.get('max_velocity', 1.0)
    plt.figure(figsize=(8, 6))

    # 1. 转置数据: RD_data.T -> 形状变为 (Range, Doppler)
    # 2. 交换 extent: [x_min, x_max, y_min, y_max] -> [doppler_min, doppler_max, range_min, range_max]
    # 3. 交换坐标轴标签
    plt.imshow(RD_data.T, aspect='auto', origin='lower', extent=[-max_vel, max_vel, 0, max_range])

    plt.colorbar(label='Amplitude')
    plt.title(f'Range-Doppler View - {i}')
    plt.xlabel('Doppler (m/s)') # X 轴是速度
    plt.ylabel('Range (m)')      # Y 轴是距离
    plt.savefig(os.path.join(root_dir, f'RD_{i}.png'))
    plt.close()

def plot_AD(AD_data, params, root_dir, i):
    """绘制并保存Angle-Doppler视图。（此函数保持不变）"""
    fov = params.get('azimuth_fov', 90.0)
    max_vel = params.get('max_velocity', 1.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(AD_data, aspect='auto', origin='lower', extent=[-fov/2, fov/2, -max_vel, max_vel])
    plt.colorbar(label='Amplitude')
    plt.title(f'Angle-Doppler View - {i}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Doppler (m/s)')
    plt.savefig(os.path.join(root_dir, f'AD_{i}.png'))
    plt.close()

# ==============================================================================
# K-Radar 数据集类
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

class K_Radar(Dataset):
    """
    用于 K-Radar 数据集的 PyTorch Dataset 类。
    """
    def __init__(self, sequences, sensor_params):
        self.sequences = sequences
        
        # 硬编码预先计算好的数据集统计值
        self.mean_log = 4.047957
        self.std_log = 0.035730
        
        # 定义统一的目标尺寸 (Doppler, Angle, Range)
        self.target_shape = (256, 256, 64)
        self.raw_params = sensor_params 
        self.ordered_keys = [
                'sensor_awr', 'sensor_retina', 'platform_mobile', 'platform_static', 'platform_mixed',
                'azimuth_fov', 'max_range', 'max_velocity', 'range_resolution',
                'velocity_resolution', 'angular_resolution', 'has_velocity'
            ]
        
        condition_vector_list  = [self.raw_params[key] for key in self.ordered_keys]
        
        normalized_vector = normalize_condition_vector_for_film(condition_vector_list, self.ordered_keys)

        self.condition_tensor = torch.tensor(normalized_vector, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        RAD_filename = self.sequences[idx]
        RAD_data = readRAD(RAD_filename)
        # 维度变化0-2
        RAD_data = np.transpose(RAD_data, (1, 2, 0))
        if RAD_data is None:
            raise ValueError(f"RAD文件未找到或加载失败: {RAD_filename}")
        
        RAD_data = preprocess_data(RAD_data)
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        RAD_data = normalize_data(RAD_data)
        
        RAD_data = resize(RAD_data, self.target_shape, anti_aliasing=True, preserve_range=True)
        
        RAD_data = RAD_data[np.newaxis, :, :, :]
        
        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'condition': self.condition_tensor, # <-- 给FiLM
            'raw_params': self.raw_params       # <-- 给PIPE
        }

# ==============================================================================
# K-Radar 数据集工厂函数
# ==============================================================================

def Create_K_Radar_Pretrain_Dataset(data_dir, sensor_config_path):
    """创建并返回一个 K-Radar 预训练数据集实例。"""
    print("-" * 60)
    print("开始加载 K-Radar 数据集...")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"K-Radar 数据目录未找到: {data_dir}")
    
    sequences_all = glob.glob(os.path.join(data_dir, "*", "*", "*", "DAR.npy"))
    config_key = "K-Radar"

    if not (sensor_config_path and os.path.exists(sensor_config_path)):
        raise FileNotFoundError(f"传感器配置文件未找到: {sensor_config_path}")
    
    with open(sensor_config_path, 'r') as f:
        sensor_configs = json.load(f)

    if config_key not in sensor_configs:
        raise KeyError(f"在配置文件中未找到配置键: '{config_key}'")
    
    sensor_params = sensor_configs[config_key]

    print(f"已加载 '{config_key}' 配置。找到 {len(sequences_all)} 个文件。")

    return K_Radar(sequences_all, sensor_params=sensor_params)

# ==============================================================================
# 主执行块: 用于验证和随机可视化
# ==============================================================================
if __name__ == '__main__':
    # --- 用户配置区域 ---
    K_RADAR_DATA_DIR = '/mnt/truenas_datasets/Datasets_mmRadar/K-Radar_reprocessed_HL_DPFT'
    SENSOR_CONFIG_FILE = "/media/ljm/Raid/ChenHongliang/RAGM/datasets/sensor_params.json"
    PLOT_OUTPUT_DIR = "./plot_outputs"
    NUM_PLOTS = 5  # 定义要随机可视化多少个样本
    # --- 配置结束 ---

    try:
        # 步骤 1: 实例化 K-Radar 数据集
        print("="*60)
        dataset = Create_K_Radar_Pretrain_Dataset(
            data_dir=K_RADAR_DATA_DIR,
            sensor_config_path=SENSOR_CONFIG_FILE
        )
        # dataset_name = dataset.dataset_params.get('dataset_name', 'Unknown')
        # print(f"成功实例化 '{dataset_name}' 数据集，大小: {len(dataset)} 个样本")
        print(f"使用的均值: {dataset.mean_log:.6f}, 标准差: {dataset.std_log:.6f}")
        print("="*60)
        
        # 步骤 2: 随机选择样本进行可视化
        if len(dataset) > 0:
            print(f"\n随机选择 {NUM_PLOTS} 个样本进行处理和可视化...")
            print(f"绘图将保存到: {os.path.abspath(PLOT_OUTPUT_DIR)}")
            
            # 从数据集中随机选择 N 个索引
            random_indices = random.sample(range(len(dataset)), k=min(NUM_PLOTS, len(dataset)))
            
            for idx in random_indices:
                print(f"\n--- 正在处理样本索引: {idx} ---")
                sample = dataset[idx]
                data_tensor = sample['data']
                # file_identifier = f"{dataset_name}_sample_{idx}"
                
                print(f"  最终数据张量形状 (C, D, A, R): {data_tensor.shape}")
                print(f"  数据范围: [{data_tensor.min():.4f}, {data_tensor.max():.4f}]")
                
        #         plot_RAD(
        #             rad_data=data_tensor,
        #             params=dataset.dataset_params,
        #             root_dir=PLOT_OUTPUT_DIR,
        #             i=file_identifier
        #         )
        #         print(f"  已为样本 {idx} 生成并保存视图。")
        # else:
            print("\n数据集为空，无法进行可视化。")

    except Exception as e:
        print(f"\n" + "!"*60)
        print(f"!!! 脚本执行过程中发生严重错误: {e}")
        print("!"*60)
        import traceback
        traceback.print_exc()