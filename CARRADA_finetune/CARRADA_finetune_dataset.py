import random
import sys
from pathlib import Path

import torch
# 清除所有模块缓存（重要！）
sys.modules.pop('datasets', None)
sys.modules.pop('datasets.CARRADA', None)
# 重新添加项目根目录（使用绝对路径）
project_root = Path(__file__).parents[2]  # 根据实际层级调整
sys.path.insert(0, str(project_root))
# 打印调试信息
# print(f"最终搜索路径：{sys.path}")
# print(f"数据集路径是否存在：{(project_root/'datasets').exists()}")
# print(f"CARRADA.py是否存在：{(project_root/'datasets'/'CARRADA.py').exists()}")
import json
import os
from pathlib import Path
from torch.utils.data import Dataset

from cv2 import transform
import numpy as np
from datasets.CARRADA import readRAD
from models.CARRADA_finetune.utils.paths import Paths
from scipy.ndimage import zoom

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

def Create_CARRADA_Finetune_Dataset(valdatatype='Validation', sensor_config_path=None):
    carrada_loader = Carrada()
    train_sequences = carrada_loader.get('Train')
    test_eval_sequences = carrada_loader.get('Test')
    # fake_sequences = carrada_loader.get('Test')

    # 加载传感器参数
    sensor_params = None
    if sensor_config_path and os.path.exists(sensor_config_path):
        with open(sensor_config_path, 'r') as f:
            sensor_configs = json.load(f)
            sensor_params = sensor_configs.get("CARRADA", {})

    train_dataset = SequenceCarradaDataset(train_sequences, is_train=True, sensor_params=sensor_params)
    validate_dataset = SequenceCarradaDataset(test_eval_sequences, sensor_params=sensor_params)
    return train_dataset, validate_dataset

class Carrada:
    """Class to load CARRADA dataset"""

    def __init__(self):
        self.paths = Paths().get()
        self.warehouse = self.paths['warehouse']
        self.carrada = self.paths['carrada']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self._split()

    def _load_data_seq_ref(self):
        path = self.carrada / 'data_seq_ref.json'
        with open(path, 'r') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.carrada / 'light_dataset_frame_oriented.json'
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        for sequence in self.annotations.keys():
            split = self.data_seq_ref[sequence]['split']
            if split == 'Train':
                self.train[sequence] = self.annotations[sequence]
            elif split == 'Validation':
                self.validation[sequence] = self.annotations[sequence]
            elif split == 'Test':
                self.test[sequence] = self.annotations[sequence]
            else:
                raise TypeError('Type {} is not supported for splits.'.format(split))

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))

import os
import numpy as np
from torch.utils.data import Dataset

class SequenceCarradaDataset(Dataset):
    def __init__(self, sequences, is_train=False, root_radar='/media/ljm/Raid/ChenHongliang/CARRADA/CARRADA',
                 root_labels='/media/ljm/Raid/ChenHongliang/CARRADA/CARRADA/Carrada', sensor_params=None):
        """
        Args:
            sequences: 字典格式，如 {'2019-09-16-12-52-12': ['000000', '000001', ...]}
            root_radar: 原始RAD数据的根目录
            root_labels: 标签数据（range-angle/range-doppler）的根目录
        """
        self.sequences = sequences
        self.is_train = is_train
        self.root_radar = root_radar
        self.root_labels = root_labels
        self.samples = []
        self.mean_log = 3.214459
        self.std_log = 0.644378
        self.range_flip_prob=0.5
        self.angle_flip_prob=0.5
        self.doppler_flip_prob=0.5

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

        # 构建有效样本列表（确保所有文件存在）
        for seq_id, frame_ids in sequences.items():
            for frame_id in frame_ids:
                # (1) 原始RAD数据路径
                rad_path = os.path.join(
                    root_radar,
                    'Carrada_RAD',
                    seq_id,
                    'RAD_numpy',
                    f"{frame_id}.npy"
                )
            
                label_ra_path = os.path.join(
                    root_labels,
                    seq_id,
                    'annotations/dense',
                    frame_id,
                    'range_angle.npy',
                )
                label_rd_path = os.path.join(
                    root_labels,
                    seq_id,
                    'annotations/dense',
                    frame_id,
                    'range_doppler.npy'
                )
                # ra_path = os.path.join(
                #     root_labels,
                #     seq_id,
                #     'range_angle_processed',
                #     frame_id + '.npy'
                # )
                # rd_path = os.path.join(
                #     root_labels,
                #     seq_id,
                #     'range_doppler_processed',
                #     frame_id + '.npy',
                # )
                # 检查所有文件是否存在
                if all(os.path.exists(p) for p in [rad_path, label_ra_path, label_rd_path]):
                    self.samples.append({
                        'rad': rad_path,
                        'label_ra': label_ra_path,  # Range-Angle标签
                        'label_rd': label_rd_path,  # Range-Doppler标签
                        'sequence': seq_id,
                        'frame_id': frame_id,
                        'condition': self.condition_tensor,
                        'params': self.raw_params
                    })
                else:
                    print(f"警告：缺少文件，跳过样本 {seq_id}/{frame_id}")

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id_str = f"{sample['sequence']}_{sample['frame_id']}"
        # 加载数据
        rad_data = np.load(sample['rad']) 
        RAD_data = preprocess_data(rad_data)
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        # normalize to [0, 1]
        RAD_data = normalize_data(RAD_data)
        
        label_ra = np.load(sample['label_ra'])
        label_rd = np.load(sample['label_rd'])

        # fake_label_ra = np.load(sample['fake_ra'])
        # fake_label_rd = np.load(sample['fake_rd'])
        
        # ra = np.load(sample['ra'])
        # rd = np.load(sample['rd'])

        
        if self.is_train:
            # 1. 随机距离翻转
            if random.random() < self.range_flip_prob:
                RAD_data = np.flip(RAD_data, axis=0).copy()       # RAD H (距离) 轴
                label_ra = np.flip(label_ra, axis=1).copy()       # label_ra H_ra (距离) 轴
                label_rd = np.flip(label_rd, axis=1).copy()       # label_rd H_rd (距离) 轴

            # 2. 随机角度翻转
            if random.random() < self.angle_flip_prob:
                RAD_data = np.flip(RAD_data, axis=1).copy()       # RAD W (角度) 轴
                label_ra = np.flip(label_ra, axis=2).copy()       # label_ra W_ra (角度) 轴
                # label_rd 通常不受角度翻转影响

            # 3. 随机多普勒翻转
            if random.random() < self.doppler_flip_prob:
                RAD_data = np.flip(RAD_data, axis=2).copy()       # RAD D (多普勒) 轴
                label_rd = np.flip(label_rd, axis=2).copy()       # label_rd D_rd (多普勒) 轴
                # label_ra 通常不受多普勒翻转影响


        frame = {'rad':  torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),
                 'rd_mask': label_rd,
                 'ra_mask': label_ra,
                 'sample_id_str': sample_id_str,
                 'sequence': sample['sequence'],
                 'frame_id': sample['frame_id'],
                 'condition': self.condition_tensor, 
                 'raw_params': self.raw_params
                 }
        return frame
    

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
    

def test():
    """Method to test the dataset"""
    # 1. 测试Carrada类的基本功能
    carrada = Carrada()
    
    # 检查数据集分割是否正确
    train_data = carrada.get('Train')
    val_data = carrada.get('Validation')
    test_data = carrada.get('Test')
    
    print(f"训练集序列数量: {len(train_data)}")
    print(f"验证集序列数量: {len(val_data)}")
    print(f"测试集序列数量: {len(test_data)}")
    
    # 2. 检查特定序列是否存在
    assert '2019-09-16-12-52-12' in train_data.keys(), "训练集缺少预期序列"
    assert '2020-02-28-13-05-44' in train_data.keys(), "训练集缺少预期序列"
    
    # 3. 测试SequenceCarradaDataset
    train_dataset = SequenceCarradaDataset(train_data)
    test_dataset = SequenceCarradaDataset(test_data)
    val_dataset = SequenceCarradaDataset(val_data)
    print(f"训练数据集长度: {len(train_dataset)}")
    print(f"训练数据集长度: {len(test_dataset)}")
    print(f"训练数据集长度: {len(val_dataset)}")
    
    # 4. 测试数据加载
    sample_idx = 0
    seq_name, seq_data = train_dataset[sample_idx]
    print(f"示例序列名称: {seq_name}")
    print(f"序列数据长度: {len(seq_data)}")
    

if __name__ == '__main__':
    test()
    print("okk")


def Create_CARRADA_Pretrain_Dataset():
    
    data_dir = '/mnt/truenas_datasets/Datasets_Radar/CARRADA/CARRADA'
    sequences_all = glob.glob(os.path.join(data_dir, "Carrada_RAD", "*/*/*.npy"))

    # sequences_all.sort()
    # sequences_all = sequences_all[:-1000]   # for test


    train_dataset = CARRADA(sequences_all)
    # test_dataset = RADDet(sequences_test, config_data)
   
    return train_dataset