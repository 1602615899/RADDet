import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob, os
import json
from scipy.ndimage import zoom

# import RADDet_loader as loader
# import RADDet_helper as helper

import time

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

class RADDet(Dataset):
    def __init__(self, sequences, config_data, sensor_params=None):
        """ Dataset for RAD sequences for pretraining """
        self.sequences = sequences
        self.config_data = config_data

        self.mean_log = 3.243838
        self.std_log = 0.643712

        self.ordered_keys = [
            'sensor_awr', 'sensor_retina', 'platform_mobile', 'platform_static', 'platform_mixed',
            'azimuth_fov', 'max_range', 'max_velocity', 'range_resolution',
            'velocity_resolution', 'angular_resolution', 'has_velocity'
        ]
        self.raw_params = sensor_params
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
        RAD_complex = readRAD(RAD_filename)
        
        if RAD_complex is None:
            raise ValueError("RAD file not found, please double check the path")
        
        # 转换为2通道数据
        RAD_data = complexTo2Channels(RAD_complex)

        # 标准化
        RAD_data = (RAD_data - self.mean_log) / self.std_log

        # 归一化到 [0, 1]
        RAD_data = normalize_data(RAD_data)    
        RAD_data = RAD_data[np.newaxis, :, :, :]

        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'condition': self.condition_tensor, # The normalized tensor for the model
            'raw_params': self.raw_params       # The original dict for visualization & pos encoding
        }
    
    def calculate_full_dataset_stats(self):
        """
        计算完整RADDet数据集的统计值
        """
        print(f"计算RADDet数据集统计值 (共 {len(self.sequences)} 个文件)")
        
        sum_val = 0.0
        sum_square = 0.0
        total_count = 0
        processed_files = 0
        
        for rad_file in tqdm(self.sequences, desc="处理RADDet文件"):
            try:
                # 读取数据
                RAD_complex = readRAD(rad_file)
                if RAD_complex is None:
                    continue
                
                # 转换为2通道数据
                RAD_data = complexTo2Channels(RAD_complex)
                
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

            print(f"RADDet数据集统计值:")
            print(f"  总处理元素数: {total_count}")
            print(f"  均值: {mean_val:.6f}")
            print(f"  标准差: {std_val:.6f}")
            
            return mean_val, std_val
        else:
            print("未能计算RADDet统计值")
            return self.mean_log, self.std_log
        
def Create_RADDet_Pretrain_Dataset(config_file_name, sensor_config_path):
    """ Read the configure file (json). """
    with open(config_file_name) as json_file:
        config = json.load(json_file)

    config_data = config["DATA"]
    sequences_train = glob.glob(os.path.join(config_data["train_set_dir"], "RAD/*/*.npy"))
    sequences_test = glob.glob(os.path.join(config_data["test_set_dir"], "RAD/*/*.npy"))
    sequences_all = sequences_train + sequences_test
    
    # 加载传感器参数
    sensor_params = None
    if sensor_config_path and os.path.exists(sensor_config_path):
        with open(sensor_config_path, 'r') as f:
            sensor_configs = json.load(f)
            sensor_params = sensor_configs.get("RADDet", {})
    
    all_dataset = RADDet(sequences_all, config_data, sensor_params=sensor_params)
    return all_dataset



####################################################################################################

def random_3d_augment(data, scale_range):
    """三维数据增强核心方法"""
    # 输入数据形状: (H=256, W=256, D=64)
    h, w, d = data.shape
        
        # scale = random.uniform(*self.scale_range)
    scale_factors = (
        random.uniform(*scale_range),
        random.uniform(*scale_range),
        random.uniform(*scale_range)
    )

    # 计算新尺寸（四舍五入保持尺寸精度）
    new_h = max(1, int(np.round(h * scale_factors[0])))
    new_w = max(1, int(np.round(w * scale_factors[1])))
    new_d = max(1, int(np.round(d * scale_factors[2])))
        
    # 重新计算精确缩放因子
    exact_scale = (
        new_h / h,
        new_w / w,
        new_d / d
    )
              
    scaled_data = np.zeros((new_h, new_w, new_d))
          
    # 执行三维插值
    zoomed = zoom(
        data, 
        exact_scale,
        order=1,
        mode='nearest'
    )

    # 严格形状校验
    assert zoomed.shape == (new_h, new_w, new_d), \
            f"缩放形状错误: 预期{(new_h, new_w, new_d)} 实际{zoomed.shape}"
    scaled_data = zoomed
        
    # 三维随机裁剪（基于实际缩放尺寸）
    target_h, target_w, target_d, = data.shape  # (64, 256, 256)
    crop_h = min(new_h, target_h)
    crop_w = min(new_w, target_w)
    crop_d = min(new_d, target_d)
        
    start_h = random.randint(0, max(0, new_h - crop_h))
    start_w = random.randint(0, max(0, new_w - crop_w))
    start_d = random.randint(0, max(0, new_d - crop_d))
        
        # 执行裁剪
    cropped = scaled_data[
        start_h:start_h + crop_h,
        start_w:start_w + crop_w,
        start_d:start_d + crop_d
    ]
        
    # 三维中心填充
    padded = np.zeros((target_h, target_w, target_d))
    pad_h = (target_h - crop_h) // 2
    pad_w = (target_w - crop_w) // 2
    pad_d = (target_d - crop_d) // 2
        
    padded[
        pad_h:pad_h + crop_h,
        pad_w:pad_w + crop_w,
        pad_d:pad_d + crop_d
    ] = cropped
        
    # 保存变换参数
    transform_params = {
        'scale_factors': exact_scale,
        'crop_starts': (start_h, start_w, start_d),
        'crop_size': (crop_h, crop_w, crop_d),
        'pads': (pad_h, pad_w, pad_d)
    }
        
    return padded, transform_params

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



def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array, scalar=1., log_10=True)      # 10log10 的话，均值方差会不会就变了？
    # output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    # target_array = np.concatenate([target_array.real, target_array.imag],axis=3)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array


# 测试代码
if __name__ == '__main__':
    # 测试RADDet数据集
    config_file_name = '/media/ljm/Raid/ChenHongliang/RAGM/datasets/RADDet_config.json'  # 请修改为实际路径
    try:
        dataset_train = Create_RADDet_Pretrain_Dataset(config_file_name)
        print(f"RADDet数据集大小: {len(dataset_train)}")
        
        # 计算完整数据集统计值
        print('\n=== 计算RADDet完整数据集统计值 ===')
        mean_val, std_val = dataset_train.calculate_full_dataset_stats()
        print(f"RADDet完整数据集统计值:")
        print(f"  均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        
        # 更新数据集的统计值
        dataset_train.mean_log = mean_val
        dataset_train.std_log = std_val
        print("已更新RADDet数据集的统计值")
        
    except Exception as e:
        print(f"RADDet数据集测试失败: {e}")
        import traceback
        traceback.print_exc()