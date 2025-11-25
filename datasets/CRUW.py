# 在脚本的开头部分，确保有这些导入
import json
import random
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset # DataLoader 也需要导入
import numpy as np
import os, glob
# from torch.utils import data as torch_data # 这行你已经有了，很好
import pickle
from tqdm import tqdm
from importlib import import_module
from scipy.ndimage import zoom
import math
import scipy.constants
import yaml
from multiprocessing import Pool, cpu_count # <--- 新增导入
import traceback # 用于打印完整的错误信息 (如果需要)
from skimage.transform import resize


def Create_CRUW_Pretrain_Dataset(CRUW_config_path, sensor_config_path=None):
    # 1. 加载配置文件
    config_dict = load_configs_from_file(CRUW_config_path)

    # 2. 初始化 CRUW 元数据对象
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], 
                   sensor_config_name='/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/cruw/dataset_configs/sensor_config_rod2021.json',
                   object_config_name='/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/cruw/dataset_configs/object_config.json')
    
    # 3. 加载传感器参数
    sensor_params = None
    if sensor_config_path and os.path.exists(sensor_config_path):
        with open(sensor_config_path, 'r') as f:
            sensor_configs = json.load(f)
            sensor_params = sensor_configs.get("CRUW", {})
    
    # 4. 指定预处理后的 .pkl 文件目录
    data_dir = '/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/data'
    
    # 5. 创建训练集实例，传入传感器参数
    train_dataset = Pretrain_CRDataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, split='train', sensor_params=sensor_params)
    
    # 6. 创建验证/测试集实例，传入传感器参数
    validate_dataset = Pretrain_CRDataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, split='test', sensor_params=sensor_params)
    
    # 7. 返回数据集列表
    dataset_all = []
    dataset_all.append(train_dataset)
    dataset_all.append(validate_dataset)
    dataset_train = ConcatDataset(dataset_all)
    
    return dataset_train
    
# ==============================================================================
# >> 辅助类和函数定义 <<
# ==============================================================================

class SensorConfig:
    """ SensorConfig 类，用于指定数据集传感器设置。 """
    def __init__(self, dataset: str, camera_cfg: dict, radar_cfg: dict, calib_cfg: dict):
        self.dataset = dataset
        self.camera_cfg = camera_cfg
        self.radar_cfg = radar_cfg
        self.calib_cfg = calib_cfg

    @classmethod
    def initialize(cls, content: dict):
        return cls(
            content.get("dataset", ""),
            content.get("camera_cfg", {}),
            content.get("radar_cfg", {}),
            content.get("calib_cfg", {})
        )

    def load_cam_calibs(self, data_root, calib_yaml_paths):
        """加载相机标定文件 (YAML格式)"""
        self.calib_cfg['cam_calib'] = {}
        self.calib_cfg['cam_calib']['load_success'] = True # 默认为True
        if not calib_yaml_paths or not isinstance(calib_yaml_paths, dict):
            self.calib_cfg['cam_calib']['load_success'] = False
            return

        for date_key, paths_for_date in calib_yaml_paths.items():
            self.calib_cfg['cam_calib'][date_key] = {}
            if not isinstance(paths_for_date, list) or not paths_for_date:
                continue
            
            cam_names = ['cam_0', 'cam_1'] 
            for i, cam_path_suffix in enumerate(paths_for_date):
                if i >= len(cam_names): break 
                
                calib_yaml_path = os.path.join(data_root, cam_path_suffix)
                cam_id_str = cam_names[i]
                self.calib_cfg['cam_calib'][date_key][cam_id_str] = {}

                if os.path.exists(calib_yaml_path):
                    try:
                        with open(calib_yaml_path, "r") as stream:
                            data_loaded = yaml.safe_load(stream)
                        K, D, R, P = parse_cam_matrices(data_loaded)
                        self.calib_cfg['cam_calib'][date_key][cam_id_str]['camera_matrix'] = K
                        self.calib_cfg['cam_calib'][date_key][cam_id_str]['distortion_coefficients'] = D
                        self.calib_cfg['cam_calib'][date_key][cam_id_str]['rectification_matrix'] = R
                        self.calib_cfg['cam_calib'][date_key][cam_id_str]['projection_matrix'] = P
                    except Exception as e:
                        print(f"警告: 加载或解析相机标定文件 {calib_yaml_path} 失败: {e}")
                        self.calib_cfg['cam_calib']['load_success'] = False
                else:
                    self.calib_cfg['cam_calib']['load_success'] = False
class ObjectConfig:
    """ ObjectConfig 类，用于指定数据集中的物体配置。 """
    def __init__(self, n_class: int, classes: list, sizes: dict):
        self.n_class = n_class # 使用 n_class
        self.classes = classes
        self.sizes = sizes

    @classmethod
    def initialize(cls, content: dict):
        # 您的配置文件中用的是 n_class, 假设 initialize 也能处理或优先用 n_class
        return cls(
            content.get("n_class", content.get("n_classes", 0)), # 兼容 n_class 和 n_classes
            content.get("classes", []),
            content.get("sizes", {})
        )

def parse_cam_matrices(data_in_parse): # 变量名保持与之前脚本一致
    """从加载的YAML数据中解析相机矩阵"""
    camera_matrix_data = data_in_parse['camera_matrix']
    distortion_coefficients_data = data_in_parse['distortion_coefficients']
    rectification_matrix_data = data_in_parse['rectification_matrix']
    projection_matrix_data = data_in_parse['projection_matrix']

    K = np.array(camera_matrix_data['data']).reshape((camera_matrix_data['rows'], camera_matrix_data['cols']))
    D = np.array(distortion_coefficients_data['data']).reshape((distortion_coefficients_data['rows'], distortion_coefficients_data['cols'])).squeeze()
    R = np.array(rectification_matrix_data['data']).reshape((rectification_matrix_data['rows'], rectification_matrix_data['cols']))
    P = np.array(projection_matrix_data['data']).reshape((projection_matrix_data['rows'], projection_matrix_data['cols']))
    return K, D, R, P

def confmap2ra(radar_configs, name, radordeg='rad'):
    Fs = radar_configs.get('sample_freq', 2e6) 
    sweepSlope = radar_configs.get('sweep_slope', 35e12) 
    num_crop = radar_configs.get('crop_num', 0) 
    ramap_rsize = radar_configs.get('ramap_rsize', 128)
    ramap_asize = radar_configs.get('ramap_asize', 128)
    fft_Rang_total = ramap_rsize + 2 * num_crop 
    c = scipy.constants.speed_of_light 
    if name == 'range':
        freq_res = Fs / fft_Rang_total; freq_grid = np.arange(fft_Rang_total) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        return rng_grid[num_crop:(fft_Rang_total - num_crop)] 
    if name == 'angle':
        ra_min_deg = radar_configs.get('ra_min', -60.0); ra_max_deg = radar_configs.get('ra_max', 60.0)
        w = np.linspace(math.sin(math.radians(ra_min_deg)), math.sin(math.radians(ra_max_deg)), ramap_asize)
        agl_grid_rad = np.arcsin(w)
        return np.degrees(agl_grid_rad) if radordeg == 'deg' else agl_grid_rad
    return np.array([])

def labelmap2ra(radar_configs, name, radordeg='rad'):
    Fs = radar_configs.get('sample_freq', 2e6); sweepSlope = radar_configs.get('sweep_slope', 35e12)
    num_crop = radar_configs.get('crop_num', 0)
    ramap_rsize_label = radar_configs.get('ramap_rsize_label', radar_configs.get('ramap_rsize', 128))
    ramap_asize_label = radar_configs.get('ramap_asize_label', radar_configs.get('ramap_asize', 128))
    fft_Rang_total = ramap_rsize_label + 2 * num_crop; c = scipy.constants.speed_of_light
    if name == 'range':
        freq_res = Fs / fft_Rang_total; freq_grid = np.arange(fft_Rang_total) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        return np.flip(rng_grid[num_crop:(fft_Rang_total - num_crop)])
    if name == 'angle':
        ra_min_label = radar_configs.get('ra_min_label', radar_configs.get('ra_min', -60.0))
        ra_max_label = radar_configs.get('ra_max_label', radar_configs.get('ra_max', 60.0))
        if radordeg == 'rad': return np.linspace(math.radians(ra_min_label), math.radians(ra_max_label), ramap_asize_label)
        if radordeg == 'deg': return np.linspace(ra_min_label, ra_max_label, ramap_asize_label)
    return np.array([])

class CRUW:
    def __init__(self, data_root: str, sensor_config_name: str, object_config_name: str):
        self.data_root = data_root
        if not os.path.isabs(sensor_config_name):
            print(f"提示: sensor_config_name '{sensor_config_name}' 可能是相对路径。")
        if not os.path.isabs(object_config_name):
            print(f"提示: object_config_name '{object_config_name}' 可能是相对路径。")

        self.sensor_cfg = self._load_sensor_config(sensor_config_name)
        self.dataset = self.sensor_cfg.dataset 
        
        if not os.path.exists(object_config_name):
            print(f"警告: 物体配置文件 '{object_config_name}' 未找到。将使用虚拟ObjectConfig。")
            # 使用配置文件中的n_class，如果可用
            n_class_from_sensor_cfg = 0
            if self.sensor_cfg and self.sensor_cfg.radar_cfg and isinstance(self.sensor_cfg.radar_cfg.get('object_cfg'), dict):
                 n_class_from_sensor_cfg = self.sensor_cfg.radar_cfg.get('object_cfg').get('n_class', 3) # 假设默认3类
            
            dummy_obj_data = {"n_class": n_class_from_sensor_cfg, "classes": [], "sizes": {}}
            self.object_cfg = ObjectConfig.initialize(dummy_obj_data)
        else:
            self.object_cfg = self._load_object_config(object_config_name)

        if self.sensor_cfg and self.sensor_cfg.radar_cfg:
            self.range_grid = confmap2ra(self.sensor_cfg.radar_cfg, name='range')
            self.angle_grid = confmap2ra(self.sensor_cfg.radar_cfg, name='angle')
            self.range_grid_label = labelmap2ra(self.sensor_cfg.radar_cfg, name='range')
            self.angle_grid_label = labelmap2ra(self.sensor_cfg.radar_cfg, name='angle')
        else: 
            print("警告: CRUW sensor_cfg.radar_cfg 未定义，无法初始化网格。")
            self.range_grid, self.angle_grid, self.range_grid_label, self.angle_grid_label = [], [], [], []

    def _load_sensor_config(self, config_name) -> SensorConfig:
        if not os.path.exists(config_name): raise FileNotFoundError(f"传感器配置文件未找到: {config_name}")
        with open(config_name, 'r') as f: data_loaded = json.load(f)
        cfg = SensorConfig.initialize(data_loaded)
        if cfg.calib_cfg and cfg.calib_cfg.get('cam_calib_paths'):
            cfg.load_cam_calibs(self.data_root, cfg.calib_cfg['cam_calib_paths'])
        return cfg

    def _load_object_config(self, config_name) -> ObjectConfig:
        if not os.path.exists(config_name): raise FileNotFoundError(f"物体配置文件未找到: {config_name}")
        with open(config_name, 'r') as f: data_loaded = json.load(f)
        # 确保 initialize 接收到它期望的键（n_class 或 n_classes）
        if "n_class" in data_loaded and "n_classes" not in data_loaded:
             # ObjectConfig.initialize 使用 .get("n_classes", 0)，所以如果配置文件是 n_class，需要转换
             data_loaded["n_classes"] = data_loaded.pop("n_class")
        return ObjectConfig.initialize(data_loaded)


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


from torch.utils.data import Dataset, DataLoader
class Pretrain_CRDataset(Dataset):
    """ 用于预训练的 CRUW 数据集加载器 """
    def __init__(self, data_dir, dataset: CRUW, config_dict: dict, split: str, 
                 is_random_chirp=True, subset=None, sensor_params=None):
        self.mean_log = 0.001544
        self.std_log = 0.017349

        self.raw_params = sensor_params
        
        self.ordered_keys = [
            'sensor_awr', 'sensor_retina', 'platform_mobile', 'platform_static', 'platform_mixed',
            'azimuth_fov', 'max_range', 'max_velocity', 'range_resolution',
            'velocity_resolution', 'angular_resolution', 'has_velocity'
        ]
        
        # 从原始参数创建向量
        condition_vector_list  = [self.raw_params[key] for key in self.ordered_keys]
        
        # 应用标准 [0, 1] 归一化
        normalized_vector = normalize_condition_vector_for_film(condition_vector_list, self.ordered_keys)

        # 存储最终的FiLM条件张量
        self.condition_tensor = torch.tensor(normalized_vector, dtype=torch.float32)

        self.data_dir = data_dir      
        self.dataset_meta = dataset  # 使用 dataset_meta 存储 CRUW 实例
        self.config_dict = config_dict
        self.split = split
        
        train_cfg = self.config_dict.get('train_cfg', {})
        test_cfg = self.config_dict.get('test_cfg', {})
        model_cfg = self.config_dict.get('model_cfg', {})
        sensor_radar_cfg = self.dataset_meta.sensor_cfg.radar_cfg if self.dataset_meta.sensor_cfg else {}

        self.win_size = train_cfg.get('win_size', 16) 

        if split == 'train' or split == 'valid':
            self.step = train_cfg.get('train_step', 1)
            self.stride = train_cfg.get('train_stride', 1)
        else: 
            self.step = test_cfg.get('test_step', 1)
            self.stride = test_cfg.get('test_stride', 1)
        
        self.is_random_chirp = is_random_chirp
        self.chirp_ids = sensor_radar_cfg.get('chirp_ids', [0,64,128,192]) 
        
        if 'mnet_cfg' in model_cfg and model_cfg['mnet_cfg']:
            self.n_chirps_to_load = model_cfg['mnet_cfg'][0] 
            self.load_all_mnet_chirps = True 
        else:
            self.n_chirps_to_load = 1 
            self.load_all_mnet_chirps = False

        self.image_paths_list = [] 
        self.radar_paths_list = [] 
        self.index_mapping = []   
        self.n_data_samples = 0   

        if subset is not None:
            self.data_files = [subset + '.pkl']
        else:
            self.data_files = list_pkl_filenames_from_prepared(data_dir, split)
        
        self.sequence_names = [name.split('.')[0] for name in self.data_files]

        print(f"为 '{split}' 部分加载 {len(self.data_files)} 个序列的元数据...")
        for seq_idx, data_filename in enumerate(tqdm(self.data_files, desc=f"处理 '{split}' 的PKL文件")):
            pkl_file_path = os.path.join(data_dir, split, data_filename)
            try:
                with open(pkl_file_path, 'rb') as f: sequence_details = pickle.load(f)
            except Exception as e:
                print(f"警告: 加载PKL {pkl_file_path} 失败 ({e}), 跳过。")
                self.image_paths_list.append([]); self.radar_paths_list.append([])
                continue
            if not all(k in sequence_details for k in ['n_frame', 'image_paths', 'radar_paths']):
                print(f"警告: PKL {pkl_file_path} 缺少键, 跳过。")
                self.image_paths_list.append([]); self.radar_paths_list.append([])
                continue
            
            num_frames_in_seq = sequence_details['n_frame']
            self.image_paths_list.append(sequence_details['image_paths'])
            self.radar_paths_list.append(sequence_details['radar_paths'])
            
            frames_span_for_one_window = (self.win_size - 1) * self.step + 1
            if self.win_size == 0 : num_samples_in_seq = 1 if num_frames_in_seq > 0 else 0
            elif num_frames_in_seq >= frames_span_for_one_window:
                num_samples_in_seq = (num_frames_in_seq - frames_span_for_one_window) // self.stride + 1
            else: num_samples_in_seq = 0
            self.n_data_samples += num_samples_in_seq
            for sample_idx_in_seq in range(num_samples_in_seq):
                self.index_mapping.append([seq_idx, sample_idx_in_seq * self.stride])
        
        if self.n_data_samples == 0: print(f"警告: '{split}' 无有效样本。")
        else: print(f"'{split}' 加载完成, 有效样本: {self.n_data_samples}。")

    def __len__(self): return self.n_data_samples

    def __getitem__(self, index):
        if not (0 <= index < self.n_data_samples):
            raise IndexError(f"索引 {index} 超出范围 (数据集大小为 {self.n_data_samples})")

        seq_idx, start_frame_offset_in_seq = self.index_mapping[index]
        radar_frames_paths_for_sequence = self.radar_paths_list[seq_idx]

        chirps_to_load_actual_ids = []
        if self.load_all_mnet_chirps:
            if len(self.chirp_ids) < self.n_chirps_to_load:
                 print(f"警告: MNet期望加载 {self.n_chirps_to_load} 个chirps, 但总共只有 {len(self.chirp_ids)} 个可用chirp ID。将使用所有可用的。")
            chirps_to_load_actual_ids = self.chirp_ids[:self.n_chirps_to_load]
        elif self.is_random_chirp: 
            if not self.chirp_ids: raise ValueError("Chirp IDs列表为空")
            chirps_to_load_actual_ids = [random.choice(self.chirp_ids)]
        else: 
            if not self.chirp_ids: raise ValueError("Chirp IDs列表为空")
            chirps_to_load_actual_ids = [self.chirp_ids[0]]
        
        sensor_radar_cfg = self.dataset_meta.sensor_cfg.radar_cfg if self.dataset_meta.sensor_cfg else {}
        ramap_rsize = sensor_radar_cfg.get('ramap_rsize', 128)
        ramap_asize = sensor_radar_cfg.get('ramap_asize', 128)
        
        num_actually_loading_chirps = len(chirps_to_load_actual_ids)

        # 统一使用多chirp的数组结构 (win_size, chirps, r, a, 2)
        radar_npy_win = np.zeros((self.win_size, num_actually_loading_chirps, ramap_rsize, ramap_asize, 2), dtype=np.float32)

        # 加载雷达数据
        for idx_in_window, frame_offset_within_window in enumerate(range(0, self.win_size * self.step, self.step)):
            current_frame_id_in_seq = start_frame_offset_in_seq + frame_offset_within_window
            if current_frame_id_in_seq >= len(radar_frames_paths_for_sequence): continue
            paths_of_all_chirps_for_this_frame = radar_frames_paths_for_sequence[current_frame_id_in_seq]

            # 统一处理多chirp情况
            for c_idx_in_npy_win, actual_chirp_id_value in enumerate(chirps_to_load_actual_ids):
                try:
                    path_list_idx = self.chirp_ids.index(actual_chirp_id_value)
                    if path_list_idx < len(paths_of_all_chirps_for_this_frame):
                        npy_file_path = paths_of_all_chirps_for_this_frame[path_list_idx]
                        radar_npy_win[idx_in_window, c_idx_in_npy_win, :, :, :] = np.load(npy_file_path)
                except (ValueError, IndexError, TypeError): pass 
        
        # 转置数据维度为 (2, win_size, chirps, r, a)
        radar_npy_win = np.transpose(radar_npy_win, (4, 0, 1, 2, 3))

        # 计算功率数据 (实部平方 + 虚部平方) -> (win_size, chirps, r, a)
        power_data = radar_npy_win[0, ...]**2 + radar_npy_win[1, ...]**2
        RAD_data = power_data  # 保持 (win_size, chirps, r, a) 结构

        # 预处理数据（对数变换）
        RAD_data = preprocess_data(RAD_data)
        
        # reshape操作：将 (win_size, chirps, r, a) 转换为 (win_size * chirps, r, a)
        n_chirps_for_reshape = num_actually_loading_chirps
        RAD_data = RAD_data.reshape(self.win_size * n_chirps_for_reshape, ramap_rsize, ramap_asize)
        
        target_h_ra, target_w_ra = 256, 256
        if ramap_rsize == 0 or ramap_asize == 0 or RAD_data.shape[0] == 0:
            expected_depth = self.win_size * n_chirps_for_reshape
            RAD_data = np.zeros((expected_depth, target_h_ra, target_w_ra), dtype=np.float32)
        else:
            target_shape_3d = (RAD_data.shape[0], target_h_ra, target_w_ra)
            
            # 使用 resize 达到同样的效果
            RAD_data = resize(RAD_data, 
                              target_shape_3d, 
                              order=1,               # order=1 对应双线性插值，与 zoom 默认值类似
                              preserve_range=True,   # 必须设置，保持数值范围
                              anti_aliasing=False)   # 通常在升采样时关闭抗锯齿以保持锐度

        # 归一化处理
        RAD_data = (RAD_data - self.mean_log) / self.std_log
        RAD_data = normalize_data(RAD_data)
        RAD_data = np.transpose(RAD_data, (1, 2, 0))
        RAD_data = RAD_data[np.newaxis, :, :, :]
        # print("cruw:",RAD_data.shape)
        return {
            'data': torch.tensor(RAD_data, dtype=torch.float32),
            'condition': self.condition_tensor, # The normalized tensor for the model
            'raw_params': self.raw_params       # The original dict for visualization & pos encoding
        }
    
    def calculate_proper_stats(self):
        """
        正确计算CRUW数据集的统计值（不进行标准化）
        """
        print(f"正确计算CRUW数据集统计值 (共 {self.n_data_samples} 个样本)")
        
        sum_val = 0.0
        sum_square = 0.0
        total_count = 0
        processed_samples = 0
        
        # 限制样本数量以加快计算
        sample_limit = min(len(self), 1000)  # 只计算1000个样本
        
        for index in tqdm(range(sample_limit), desc="计算CRUW统计值"):
            try:
                # 手动重现数据加载过程，但不进行标准化
                seq_idx, start_frame_offset_in_seq = self.index_mapping[index]
                radar_frames_paths_for_sequence = self.radar_paths_list[seq_idx]

                # 重现chirp选择逻辑
                chirps_to_load_actual_ids = []
                if self.load_all_mnet_chirps:
                    chirps_to_load_actual_ids = self.chirp_ids[:self.n_chirps_to_load]
                elif self.is_random_chirp: 
                    chirps_to_load_actual_ids = [random.choice(self.chirp_ids)]
                else: 
                    chirps_to_load_actual_ids = [self.chirp_ids[0]]
                
                sensor_radar_cfg = self.dataset_meta.sensor_cfg.radar_cfg if self.dataset_meta.sensor_cfg else {}
                ramap_rsize = sensor_radar_cfg.get('ramap_rsize', 128)
                ramap_asize = sensor_radar_cfg.get('ramap_asize', 128)
                
                num_actually_loading_chirps = len(chirps_to_load_actual_ids)

                # 加载原始雷达数据（复数数据）
                radar_npy_win = np.zeros((self.win_size, num_actually_loading_chirps, ramap_rsize, ramap_asize, 2), dtype=np.float32)

                for idx_in_window, frame_offset_within_window in enumerate(range(0, self.win_size * self.step, self.step)):
                    current_frame_id_in_seq = start_frame_offset_in_seq + frame_offset_within_window
                    if current_frame_id_in_seq >= len(radar_frames_paths_for_sequence): continue
                    paths_of_all_chirps_for_this_frame = radar_frames_paths_for_sequence[current_frame_id_in_seq]

                    for c_idx_in_npy_win, actual_chirp_id_value in enumerate(chirps_to_load_actual_ids):
                        try:
                            path_list_idx = self.chirp_ids.index(actual_chirp_id_value)
                            if path_list_idx < len(paths_of_all_chirps_for_this_frame):
                                npy_file_path = paths_of_all_chirps_for_this_frame[path_list_idx]
                                radar_npy_win[idx_in_window, c_idx_in_npy_win, :, :, :] = np.load(npy_file_path)
                        except (ValueError, IndexError, TypeError): pass 
                
                # 转置数据维度为 (2, win_size, chirps, r, a)
                radar_npy_win = np.transpose(radar_npy_win, (4, 0, 1, 2, 3))

                # 计算功率数据 (实部平方 + 虚部平方) -> (win_size, chirps, r, a)
                power_data = radar_npy_win[0, ...]**2 + radar_npy_win[1, ...]**2
                
                # 只进行对数变换，不进行标准化
                log_power_data = preprocess_data(power_data)  # 对数变换
                
                # reshape操作
                n_chirps_for_reshape = num_actually_loading_chirps
                log_power_data = log_power_data.reshape(self.win_size * n_chirps_for_reshape, ramap_rsize, ramap_asize)
                
                # 插值到目标尺寸
                target_h_ra, target_w_ra = 256, 256
                if ramap_rsize > 0 and ramap_asize > 0:
                    zoom_h_factor = target_h_ra / ramap_rsize
                    zoom_w_factor = target_w_ra / ramap_asize
                    zoom_factors = (1, zoom_h_factor, zoom_w_factor)
                    log_power_data = zoom(log_power_data, zoom_factors, order=1)

                # 累积统计值（使用对数变换后的数据）
                sum_val += np.sum(log_power_data)
                sum_square += np.sum(log_power_data ** 2)
                total_count += log_power_data.size
                processed_samples += 1
                
            except Exception as e:
                print(f"样本 {index} 处理失败: {e}")
                continue
        
        print(f"成功处理 {processed_samples} 个样本")
        
        # 计算最终统计值
        if total_count > 0:
            mean_val = sum_val / total_count
            variance_val = (sum_square / total_count - mean_val ** 2)
            std_val = np.sqrt(max(0, variance_val))

            print(f"CRUW数据集统计值:")
            print(f"  总处理元素数: {total_count}")
            print(f"  对数变换后均值: {mean_val:.6f}")
            print(f"  对数变换后标准差: {std_val:.6f}")
            
            return mean_val, std_val
        else:
            print("未能计算CRUW统计值")
            return 0.0, 1.0  # 返回默认值

# ==============================================================================
# >> 您提供的其他预处理函数保持不变 <<
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

def preprocess_data(power_data_input): 
    """
    对输入的功率数据进行预处理，这里主要是对数变换。
    power_data_input: 已经是计算好的功率值 (例如 real^2 + imag^2)
    """
    log_power_data = getLog(power_data_input)
    return log_power_data

def getLog(target_array, scalar=1., log_10=True):
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else: 
        return scalar * np.log(target_array + 1.)

# ==============================================================================
# >> 其他辅助函数 (load_configs_from_file, list_pkl_filenames_from_prepared) <<
# ==============================================================================
def load_configs_from_file(config_py_path: str) -> dict:
    if not os.path.exists(config_py_path): raise FileNotFoundError(f"配置文件未找到: {config_py_path}")
    module_name = os.path.basename(config_py_path)[:-3]
    if '.' in module_name: raise ValueError("配置文件名中不允许包含'.'")
    config_dir = os.path.dirname(config_py_path)
    original_sys_path = list(sys.path) 
    path_inserted_flag = False
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir); path_inserted_flag = True
    try: mod = import_module(module_name)
    finally:
        if path_inserted_flag and sys.path and sys.path[0] == config_dir: sys.path.pop(0)
        elif path_inserted_flag : sys.path = original_sys_path 
    return {name: value for name, value in mod.__dict__.items() if not name.startswith('__')}

def list_pkl_filenames_from_prepared(data_dir: str, split: str) -> list:
    split_path = os.path.join(data_dir, split)
    if not os.path.isdir(split_path):
        print(f"警告: PKL数据目录未找到: {split_path}"); return []
    pkl_files = sorted([f for f in os.listdir(split_path) if f.endswith('.pkl')])
    if not pkl_files: print(f"警告: 在 {split_path} 中没有找到 .pkl 文件。")
    return pkl_files

# ==============================================================================
# >> 主要执行逻辑 (Create_And_Prepare_Datasets_For_Stats, if __name__ ...) <<
# ==============================================================================
def Create_And_Prepare_Datasets_For_Stats(
    python_config_file_path: str, 
    dataset_base_root_path: str, 
    prepared_pkl_data_path: str, 
    sensor_config_json_file_path: str,
    object_config_json_file_path: str 
    ) -> list:
    print(f"步骤 1: 从 '{python_config_file_path}' 加载主配置文件...")
    active_config_dict = load_configs_from_file(python_config_file_path)
    
    print(f"步骤 2: 使用数据根目录 '{dataset_base_root_path}' 初始化 CRUW 元数据对象...")
    cruw_meta_instance = CRUW(
        data_root=dataset_base_root_path, 
        sensor_config_name=sensor_config_json_file_path,
        object_config_name=object_config_json_file_path
    )
    datasets_for_stats = []
    print(f"步骤 3: 从 '{prepared_pkl_data_path}' 为 'train' 部分创建 Pretrain_CRDataset 实例...")
    train_dataset_instance = Pretrain_CRDataset(
        data_dir=prepared_pkl_data_path, dataset=cruw_meta_instance, 
        config_dict=active_config_dict, split='train'
    )
    if train_dataset_instance and len(train_dataset_instance) > 0: datasets_for_stats.append(train_dataset_instance)
    else: print("警告: 训练数据集为空或初始化失败。")
        
    print(f"步骤 4: 从 '{prepared_pkl_data_path}' 为 'test' 部分创建 Pretrain_CRDataset 实例...")
    test_dataset_instance = Pretrain_CRDataset(
        data_dir=prepared_pkl_data_path, dataset=cruw_meta_instance, 
        config_dict=active_config_dict, split='test'
    )
    if test_dataset_instance and len(test_dataset_instance) > 0: datasets_for_stats.append(test_dataset_instance)
    else: print("警告: 测试数据集为空或初始化失败。")
    return datasets_for_stats

# 你提供的辅助函数
def compute_batch_stats(batch_rad_data_tensor): # 参数名修改以明确
    # 确保输入是 torch.Tensor
    if not isinstance(batch_rad_data_tensor, torch.Tensor):
        # 如果批次内有多个样本，并且它们作为列表传递，这里可能需要调整
        # 但通常 DataLoader 会将它们堆叠成一个批次张量
        raise TypeError(f"compute_batch_stats 期望一个 PyTorch 张量, 但收到了 {type(batch_rad_data_tensor)}")

    # 将数据移到CPU并转换为float64以提高精度
    tensor_for_calculation = batch_rad_data_tensor.to(device='cpu', dtype=torch.float64)

    sum_val = torch.sum(tensor_for_calculation).item()
    sum_square = torch.sum(tensor_for_calculation ** 2).item()
    min_val = torch.min(tensor_for_calculation).item()
    max_val = torch.max(tensor_for_calculation).item()
    total_count = tensor_for_calculation.numel()
    return sum_val, sum_square, min_val, max_val, total_count

# 主计算函数
def calculate_mean_std_parallel(data_loader_for_stats): # 参数名修改
    print('正在使用并行处理计算 RAD 数据的均值和标准差...')
    
    # 准备并行计算
    # 使用 with Pool(...) 可以确保池在完成后正确关闭
    with Pool(processes=cpu_count()) as pool:
        results_async = []
        # 并行计算每个批次的统计信息
        for data_batch_from_loader in tqdm(data_loader_for_stats, desc="提交批次到进程池"):
            # Pretrain_CRDataset.__getitem__ 返回 (placeholder_path_info, RAD_data_tensor)
            # DataLoader 会将它们分别打包成批次
            # data_batch_from_loader 将是 [batched_placeholders, batched_RAD_data_tensors]
            # 我们只对 radar_data_tensor 部分感兴趣
            actual_rad_data_batch = data_batch_from_loader[1] 
            results_async.append(pool.apply_async(compute_batch_stats, args=(actual_rad_data_batch,)))
        
        # 获取结果 (这会阻塞直到所有任务完成)
        # print("等待所有批次统计完成...") # 可选的进度提示
        # completed_results = [res.get(timeout=180) for res in results_async] # timeout可选

        # 为了更安全地获取结果并处理可能的错误
        completed_results = []
        for i, res_async in enumerate(tqdm(results_async, desc="收集并行计算结果")):
            try:
                completed_results.append(res_async.get(timeout=300)) # 增加超时时间
            except Exception as e:
                print(f"错误：获取批次 {i} 的结果失败: {e}")
                # 你可以选择是跳过这个批次，还是引发一个错误停止整个过程
                # 这里我们选择跳过（或者你可以返回None并在下面处理）
                completed_results.append(None)


    # 汇总结果
    valid_results = [r for r in completed_results if r is not None]
    if not valid_results:
        print("错误：未能从任何批次收集到有效的统计结果。")
        return

    sum_val_total = sum(result[0] for result in valid_results)
    sum_square_total = sum(result[1] for result in valid_results)
    min_val_overall = min(result[2] for result in valid_results)
    max_val_overall = max(result[3] for result in valid_results)
    total_element_count_overall = sum(result[4] for result in valid_results)

    if total_element_count_overall == 0:
        print("错误: 总元素数量为0，无法计算统计量。")
        return

    # 计算最终的均值和标准差
    mean_val = sum_val_total / total_element_count_overall
    # 注意：对于方差计算，最好使用 E[X^2] - (E[X])^2，并确保是浮点数运算
    variance_val = (sum_square_total / total_element_count_overall) - (mean_val ** 2)
    if variance_val < 0: # 处理可能的浮点精度问题
        print(f"警告：计算得到的方差为负 ({variance_val:.8e})，已修正为0。")
        variance_val = 0.0
    std_val = variance_val ** 0.5

    print("\n--- 并行计算得到的 RAD 数据统计结果 ---")
    print(f"参与计算的总元素数量: {total_element_count_overall}")
    print(f"全局均值: {mean_val:.8f}")
    print(f"全局方差: {variance_val:.8f}")
    print(f"全局标准差: {std_val:.8f}")
    print(f"全局最小值: {min_val_overall:.8f}")
    print(f"全局最大值: {max_val_overall:.8f}")

    # 你可以将这些值用于更新 Pretrain_CRDataset 中的 self.mean_log 和 self.std_log
    # 例如： self.mean_log = mean_val
    #        self.std_log = std_val

# 在Create_CRUW_Pretrain_Dataset函数后添加测试代码
if __name__ == '__main__':
    try:
        CRUW_config_path = '/media/ljm/Raid/ChenHongliang/RAGM/models/CRUW_finetune/config_Rodnet.py'
        dataset_train = Create_CRUW_Pretrain_Dataset(CRUW_config_path)
        print(f"CRUW数据集大小: {len(dataset_train)}")
        
        # 访问内部数据集
        train_dataset = dataset_train.datasets[0]
        validate_dataset = dataset_train.datasets[1]
        
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"验证数据集大小: {len(validate_dataset)}")
        
        # 计算正确的统计值
        print('\n=== 计算正确的CRUW数据集统计值 ===')
        mean_val, std_val = train_dataset.calculate_proper_stats()
        print(f"CRUW数据集统计值:")
        print(f"  均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        
        # 更新__init__中的参数
        train_dataset.mean_log = mean_val
        train_dataset.std_log = std_val
        validate_dataset.mean_log = mean_val
        validate_dataset.std_log = std_val
        
        print(f"\n已更新CRUW数据集的统计值:")
        print(f"  train_dataset.mean_log = {mean_val:.6f}")
        print(f"  train_dataset.std_log = {std_val:.6f}")
        
    except Exception as e:
        print(f"CRUW数据集测试失败: {e}")
        import traceback
        traceback.print_exc()