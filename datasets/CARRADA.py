import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob, os
import json

from tqdm import tqdm

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

class CARRADA(Dataset):
    def __init__(self, sequences, sensor_params=None):
        """ Dataset for RAD sequences for pretraining """
        self.sequences = sequences

        self.mean_log = 3.214459
        self.std_log = 0.644378
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
        
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        RAD_filename = self.sequences[idx]
        RAD_data = readRAD(RAD_filename)
        if RAD_data is None:
            raise ValueError("RAD file not found, please double check the path")
        
        # 预处理数据
        RAD_data = preprocess_data(RAD_data)
        
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        # normalize to [0, 1]
        RAD_data = normalize_data(RAD_data)
        RAD_data = RAD_data[np.newaxis, :, :, :] 
        # print("carrada:",RAD_data.shape)
        
        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'condition': self.condition_tensor, # The normalized tensor for the model
            'raw_params': self.raw_params       # The original dict for visualization & pos encoding
        }
    
    def calculate_full_dataset_stats(self):
        """
        计算完整CARRADA数据集的统计值
        """
        print(f"计算CARRADA数据集统计值 (共 {len(self.sequences)} 个文件)")
        
        sum_val = 0.0
        sum_square = 0.0
        total_count = 0
        processed_files = 0
        
        for rad_file in tqdm(self.sequences, desc="处理CARRADA文件"):
            try:
                # 读取数据
                RAD_data = readRAD(rad_file)
                if RAD_data is None:
                    continue
                
                # 预处理数据
                RAD_data = preprocess_data(RAD_data)
                
                # 累积统计值
                sum_val += np.sum(RAD_data)
                sum_square += np.sum(RAD_data ** 2)
                total_count += RAD_data.size
                processed_files += 1
                
            except Exception as e:
                print(f"文件 {rad_file} 处理失败: {e}")
                continue
        
        print(f"成功处理 {processed_files} 个文件")
        
        # 计算最终统计值
        if total_count > 0:
            mean_val = sum_val / total_count
            variance_val = (sum_square / total_count - mean_val ** 2)
            std_val = np.sqrt(max(0, variance_val))  # 避免负数开方

            print(f"CARRADA数据集统计值:")
            print(f"  总处理元素数: {total_count}")
            print(f"  均值: {mean_val:.6f}")
            print(f"  标准差: {std_val:.6f}")
            
            return mean_val, std_val
        else:
            print("未能计算CARRADA统计值")
            return self.mean_log, self.std_log



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
    """ read input RAD matrices """
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None

def preprocess_data(target_array):
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def Create_CARRADA_Pretrain_Dataset(sensor_config_path=None):
    """ Create CARRADA pretraining dataset """
    print("加载CARRADA数据集...")
    
    # 数据路径
    data_dir = '/media/ljm/Raid/ChenHongliang/CARRADA/CARRADA'
    sequences_all = glob.glob(os.path.join(data_dir, "Carrada_RAD", "*/*/*.npy"))
    
    # 加载传感器参数
    sensor_params = None
    if sensor_config_path and os.path.exists(sensor_config_path):
        with open(sensor_config_path, 'r') as f:
            sensor_configs = json.load(f)
            sensor_params = sensor_configs.get("CARRADA", {})
    
    print(f"找到 {len(sequences_all)} 个CARRADA文件")
    return CARRADA(sequences_all, sensor_params=sensor_params)

# 测试代码
if __name__ == '__main__':
    try:
        dataset_train = Create_CARRADA_Pretrain_Dataset()
        print(f"CARRADA数据集大小: {len(dataset_train)}")
        
        # 计算完整数据集统计值
        print('\n=== 计算CARRADA完整数据集统计值 ===')
        mean_val, std_val = dataset_train.calculate_full_dataset_stats()
        print(f"CARRADA完整数据集统计值:")
        print(f"  均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        
        # 更新数据集的统计值
        dataset_train.mean_log = mean_val
        dataset_train.std_log = std_val
        print("已更新CARRADA数据集的统计值")
        
    except Exception as e:
        print(f"CARRADA数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

