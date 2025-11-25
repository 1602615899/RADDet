# -*- coding: utf-8 -*-
"""
此脚本定义了用于处理 RaDICaL 数据集四个子集
(indoor, outdoor30, outdoor60, highRes) 的 PyTorch Dataset 类。

这是用于训练和评估的最终版本，包含了：
- 为每个子集预先计算好的统计数据。
- 将数据 resize 到统一维度的功能。
- 用于验证和调试的、坐标轴已修正的可视化工具。

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
from tqdm import tqdm
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
    假设 rad_data 的维度顺序是 (Range, Angle, Doppler)。
    """
    os.makedirs(root_dir, exist_ok=True)
    data = rad_data.squeeze().cpu().numpy()
    
    # 通过对特定轴求平均值来生成2D视图
    RA_data = np.mean(data, axis=2) # 对Doppler轴(2)求平均 -> (Range, Angle)
    RD_data = np.mean(data, axis=1) # 对Angle轴(1)求平均   -> (Range, Doppler)
    AD_data = np.mean(data, axis=0) # 对Range轴(0)求平均   -> (Angle, Doppler)

    if RA:
        plot_RA(RA_data, params, root_dir, i)
    if RD:
        plot_RD(RD_data, params, root_dir, i)
    if AD:
        plot_AD(AD_data, params, root_dir, i)

def plot_RA(RA_data, params, root_dir, i):
    """
    绘制并保存Range-Angle视图。
    RA_data shape: (Range, Angle)
    imshow Y-axis: Range, X-axis: Angle
    """
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
    """
    绘制并保存Range-Doppler视图。
    RD_data shape: (Range, Doppler)
    imshow Y-axis: Range, X-axis: Doppler
    """
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
    """
    绘制并保存Angle-Doppler视图。
    AD_data shape: (Angle, Doppler)
    imshow Y-axis: Angle, X-axis: Doppler
    """
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
# RaDICaL 数据集定义
# ==============================================================================

class RaDICaLBase(Dataset):
    """RaDICaL 数据集的抽象基类。"""
    def __init__(self, sequences, sensor_params):
        if self.__class__ is RaDICaLBase:
            raise TypeError("基类 RaDICaLBase 不能被直接实例化。")
        self.sequences = sequences
        self.dataset_params = sensor_params
        self.mean_log = 0.0
        self.std_log = 1.0
        # 统一目标尺寸 (Range, Angle, Doppler)
        self.target_shape = (256, 256, 64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        RAD_filename = self.sequences[idx]
        RAD_data = readRAD(RAD_filename)
        if RAD_data is None:
            raise ValueError(f"RAD文件未找到或加载失败: {RAD_filename}")
        
        RAD_data = preprocess_data(RAD_data)
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        RAD_data = normalize_data(RAD_data)
        
        RAD_data = resize(RAD_data, self.target_shape, anti_aliasing=True, preserve_range=True)
        
        RAD_data = RAD_data[np.newaxis, :, :, :] 
        
        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'dataset_params': self.dataset_params,
        }

class RaDICaL_indoor(RaDICaLBase):
    def __init__(self, sequences, sensor_params):
        super().__init__(sequences, sensor_params)
        self.mean_log = 5.113496
        self.std_log = 0.762974

class RaDICaL_outdoor30(RaDICaLBase):
    def __init__(self, sequences, sensor_params):
        super().__init__(sequences, sensor_params)
        self.mean_log = 5.113455
        self.std_log = 0.762981

class RaDICaL_outdoor60(RaDICaLBase):
    def __init__(self, sequences, sensor_params):
        super().__init__(sequences, sensor_params)
        self.mean_log = 5.113440
        self.std_log = 0.762970

class RaDICaL_highRes(RaDICaLBase):
    def __init__(self, sequences, sensor_params):
        super().__init__(sequences, sensor_params)
        self.mean_log = 5.113494
        self.std_log = 0.762990

def _create_radical_subset(DatasetClass, dir_name, config_key, base_data_dir, sensor_config_path):
    print("-" * 60)
    print(f"开始加载 RaDICaL 子集: {config_key}...")
    data_path = os.path.join(base_data_dir, dir_name, 'rad_data')
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"数据目录未找到: {data_path}")
    sequences_all = glob.glob(os.path.join(data_path, "*.npy"))
    if not (sensor_config_path and os.path.exists(sensor_config_path)):
        raise FileNotFoundError(f"传感器配置文件未找到: {sensor_config_path}")
    with open(sensor_config_path, 'r') as f:
        sensor_configs = json.load(f)
    if config_key not in sensor_configs:
        raise KeyError(f"在配置文件中未找到配置键: '{config_key}'")
    sensor_params = sensor_configs[config_key]
    print(f"已加载 '{config_key}' 配置。找到 {len(sequences_all)} 个文件。")
    return DatasetClass(sequences_all, sensor_params=sensor_params)

def Create_RaDICaL_indoor_Dataset(base_data_dir, sensor_config_path):
    return _create_radical_subset(RaDICaL_indoor, 'indoor_rad_output', 'RaDICaL_indoor', base_data_dir, sensor_config_path)

def Create_RaDICaL_outdoor30_Dataset(base_data_dir, sensor_config_path):
    return _create_radical_subset(RaDICaL_outdoor30, '30m_rad_output', 'RaDICaL_outdoor30', base_data_dir, sensor_config_path)

def Create_RaDICaL_outdoor60_Dataset(base_data_dir, sensor_config_path):
    return _create_radical_subset(RaDICaL_outdoor60, '50m_rad_output', 'RaDICaL_outdoor60', base_data_dir, sensor_config_path)

def Create_RaDICaL_highRes_Dataset(base_data_dir, sensor_config_path):
    return _create_radical_subset(RaDICaL_highRes, 'highres_rad_output', 'RaDICaL_highRes', base_data_dir, sensor_config_path)

# ==============================================================================
# 主执行块: 用于验证和可视化
# ==============================================================================
if __name__ == '__main__':
    # --- 用户配置区域 ---
    RADICAL_BASE_DIR = "/media/ljm/Raid/ChenHongliang/RaDICaL-main/RaDICaL-main/rad_batch_output"
    SENSOR_CONFIG_FILE = "/media/ljm/Raid/ChenHongliang/RAGM/datasets/sensor_params.json"
    PLOT_OUTPUT_DIR = "./plot_outputs"
    # --- 配置结束 ---

    # 将所有 RaDICaL 数据集的创建信息打包
    datasets_to_process = [
        {'creator': Create_RaDICaL_indoor_Dataset, 'args': [RADICAL_BASE_DIR, SENSOR_CONFIG_FILE]},
        {'creator': Create_RaDICaL_outdoor30_Dataset, 'args': [RADICAL_BASE_DIR, SENSOR_CONFIG_FILE]},
        {'creator': Create_RaDICaL_outdoor60_Dataset, 'args': [RADICAL_BASE_DIR, SENSOR_CONFIG_FILE]},
        {'creator': Create_RaDICaL_highRes_Dataset, 'args': [RADICAL_BASE_DIR, SENSOR_CONFIG_FILE]},
    ]
    
    try:
        print("="*60)
        print("开始对所有 RaDICaL 数据集进行最终验证并生成可视化... ")
        print(f"绘图将保存到: {os.path.abspath(PLOT_OUTPUT_DIR)}")
        print("="*60)
        
        for item in datasets_to_process:
            creator = item['creator']
            args = item['args']
            
            # 1. 实例化数据集
            dataset = creator(*args)
            dataset_name = dataset.dataset_params.get('dataset_name', type(dataset).__name__)
            file_identifier = dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            print(f"\n" + "-"*60)
            print(f"正在验证: '{dataset_name}'")
            print(f"样本数量: {len(dataset)}")
            print(f"使用的均值: {dataset.mean_log:.4f}, 标准差: {dataset.std_log:.4f}")
            
            if len(dataset) > 0:
                # 2. 验证数据加载
                print("--- 验证数据加载 (__getitem__) ---")
                sample = dataset[0]
                data_tensor = sample['data']
                
                print("成功加载并处理第一个样本!")
                print(f"  最终数据张量形状 (C, R, A, D): {data_tensor.shape}")
                print(f"  数据范围: [{data_tensor.min():.4f}, {data_tensor.max():.4f}]")
                
                # 3. 可视化
                print(f"  正在生成并保存 RA, RD, AD 视图...")
                plot_RAD(
                    rad_data=data_tensor,
                    params=dataset.dataset_params,
                    root_dir=PLOT_OUTPUT_DIR,
                    i=file_identifier
                )
                print("  可视化完成。")
            else:
                print("数据集为空，跳过验证。")

    except Exception as e:
        print(f"\n" + "!"*60)
        print(f"!!! 数据集验证过程中发生错误: {e}")
        print("!"*60)
        import traceback
        traceback.print_exc()