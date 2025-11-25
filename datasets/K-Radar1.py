# -*- coding: utf-8 -*-
"""
此脚本定义了用于处理 K-Radar 数据集的 PyTorch Dataset 类。
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import json
from tqdm import tqdm

# ==============================================================================
# 模块级辅助函数 (可与其他数据集共享)
# ==============================================================================

def normalize_data(data):
    """将输入数据归一化到 [-1, 1] 的范围。"""
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min != 0:
        normalized_data = (data - data_min) / (data_max - data_min) * 2 - 1
    else:
        normalized_data = np.zeros_like(data)
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
    """对原始RAD数据进行预处理: 复数 -> 功率 -> 对数尺度。"""
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

# ==============================================================================
# K-Radar 数据集类
# ==============================================================================

class K_Radar(Dataset):
    """
    用于 K-Radar 数据集的 PyTorch Dataset 类。
    """
    def __init__(self, sequences, sensor_params):
        """
        初始化 K_Radar 数据集实例。
        
        Args:
            sequences (list): 数据文件路径的列表。
            sensor_params (dict): 包含传感器规格、数据集名称和ID的字典。
        """
        self.sequences = sequences
        self.dataset_params = sensor_params
        
        # ==================================================================
        # 重要: 以下是预先计算好的数据集统计值。
        # 这些值是通过运行一次本脚本的 `if __name__ == '__main__':` 部分中的
        # `calculate_full_dataset_stats()` 方法得到的。
        # 在正常的训练和评估流程中，我们直接使用这些硬编码的值。
        self.mean_log = 3.214459
        self.std_log = 0.644378
        # ==================================================================
    
    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        根据索引加载、预处理并返回一个数据样本。
        
        处理流程:
        1. 读取原始复数数据。
        2. 预处理 (preprocess_data): 复数 -> 功率 -> 对数尺度。
        3. 标准化 (Standardization): 使用预先计算的全局均值和标准差进行 z-score 标准化。
        4. 归一化 (Normalization): 将数据缩放到 [-1, 1] 范围。
        5. 增加通道维度。
        """
        RAD_filename = self.sequences[idx]
        RAD_data = readRAD(RAD_filename)
        if RAD_data is None:
            raise ValueError(f"RAD文件未找到或加载失败: {RAD_filename}")
        
        # 步骤 2: 预处理
        RAD_data = preprocess_data(RAD_data)
        
        # 步骤 3: 标准化 (使用类中预存的全局统计值)
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        
        # 步骤 4: 归一化
        RAD_data = normalize_data(RAD_data)
        
        # 步骤 5: 增加通道维度
        RAD_data = RAD_data[np.newaxis, :, :, :]
        
        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'dataset_id': self.dataset_params.get('dataset_id', -1),
            'dataset_params': self.dataset_params,
            'dataset_name': self.dataset_params.get('dataset_name', 'Unknown')
        }
    
    def calculate_full_dataset_stats(self):
        """
        【一次性工具】计算并返回当前 K-Radar 数据集的均值和标准差。
        此方法仅用于预处理阶段，以获取需要硬编码到 __init__ 中的统计值。
        """
        dataset_name = self.dataset_params.get('dataset_name', self.__class__.__name__)
        print(f"计算 {dataset_name} 数据集统计值 (共 {len(self.sequences)} 个文件)...")
        sum_val, sum_square, total_count, processed_files = 0.0, 0.0, 0, 0
        
        for rad_file in tqdm(self.sequences, desc=f"处理 {dataset_name} 文件"):
            try:
                # 注意：这里只进行预处理，不进行标准化或归一化
                RAD_data = preprocess_data(readRAD(rad_file))
                sum_val += np.sum(RAD_data)
                sum_square += np.sum(RAD_data ** 2)
                total_count += RAD_data.size
                processed_files += 1
            except Exception as e:
                print(f"文件 {rad_file} 处理失败: {e}")
                continue
                
        if processed_files == 0:
            print("警告: 未能成功处理任何文件，无法计算统计值。")
            return self.mean_log, self.std_log
            
        print(f"成功处理 {processed_files} 个文件。")
        
        if total_count > 0:
            mean_val = sum_val / total_count
            variance_val = (sum_square / total_count) - (mean_val ** 2)
            std_val = np.sqrt(max(0, variance_val))
            print(f"\n{dataset_name} 数据集统计值计算完成:")
            print(f"  均值 (mean_log): {mean_val:.6f}")
            print(f"  标准差 (std_log): {std_val:.6f}")
            return mean_val, std_val
        else:
            print(f"未能计算 {dataset_name} 统计值，总元素数为0。")
            return self.mean_log, self.std_log

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
    print(f"已成功加载 '{config_key}' 的配置。找到 {len(sequences_all)} 个文件。")
    return K_Radar(sequences_all, sensor_params=sensor_params)

# ==============================================================================
# 主执行块: 用于测试和一次性统计计算
# ==============================================================================
if __name__ == '__main__':
    # --- 用户配置区域 ---
    K_RADAR_DATA_DIR = '/mnt/truenas_datasets/Datasets_mmRadar/K-Radar_reprocessed_HL_DPFT'
    SENSOR_CONFIG_FILE = "/media/ljm/Raid/ChenHongliang/RAGM/datasets/sensor_params.json"
    # --- 配置结束 ---

    try:
        # 步骤 1: 实例化 K-Radar 数据集
        print("="*60)
        dataset_train = Create_K_Radar_Pretrain_Dataset(
            data_dir=K_RADAR_DATA_DIR,
            sensor_config_path=SENSOR_CONFIG_FILE
        )
        dataset_name = dataset_train.dataset_params.get('dataset_name', 'Unknown')
        print(f"成功实例化 '{dataset_name}' 数据集，大小: {len(dataset_train)} 个样本")
        print("="*60)
        
        # 步骤 2: 计算并验证数据集的统计数据 (一次性操作)
        # 运行此部分以获取 mean_log 和 std_log 的值。
        print("\n" + "="*60)
        print("执行一次性统计数据计算...")
        mean_val, std_val = dataset_train.calculate_full_dataset_stats()
        print("\n" + "="*60)
        print("重要提示: 计算完成后，请将上面打印出的 '均值 (mean_log)' 和 '标准差 (std_log)' 的值，")
        print("复制回 K_Radar 类的 __init__ 方法中，替换掉现有的硬编码值。")
        print("="*60)
        
        # 步骤 3: 验证 __getitem__ 是否正常工作
        # 这一步假设 __init__ 中的统计值已经是正确的（或者是刚刚计算出的）
        if len(dataset_train) > 0:
            print("\n--- 验证数据加载 (__getitem__) ---")
            # 更新实例的统计值，以确保本次测试使用的是刚计算出的新值
            dataset_train.mean_log = mean_val
            dataset_train.std_log = std_val
            print("已将刚计算出的统计值更新到当前数据集实例中进行测试。")
            try:
                sample = dataset_train[0]
                print("成功加载第一个样本!")
                print(f"  样本数据范围: [{sample['data'].min():.4f}, {sample['data'].max():.4f}]")
                assert -1.001 <= sample['data'].min() <= 0.001, "归一化后最小值不在[-1, 0]附近"
                assert 0.999 <= sample['data'].max() <= 1.001, "归一化后最大值不接近1"
                print("  数据范围验证通过 (接近 [-1, 1])。")
                print(f"  数据集名称: {sample['dataset_name']}")
            except Exception as e:
                print(f"!!! 加载样本或验证时出错: {e}")

    except Exception as e:
        print(f"\n" + "!"*60)
        print(f"!!! 脚本执行过程中发生严重错误: {e}")
        print("!"*60)
        import traceback
        traceback.print_exc()