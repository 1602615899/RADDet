import torch
import torch.nn.functional as F
from torch import nn

# ================== 元数据常量定义 ==================

# 1. 数据集ID映射 (作为主键)
DATASET_TO_ID = {
    'CRUW': 0, 'RADDet': 1, 'CARRADA': 2, 'RaDICaL_indoor': 3,
    'RaDICaL_outdoor30': 4, 'RaDICaL_outdoor60': 5,
    'RaDICaL_highRes': 6, 'SCORP': 7
}
ID_TO_DATASET = {v: k for k, v in DATASET_TO_ID.items()}
NUM_DATASETS = len(DATASET_TO_ID)

# 2. 传感器型号映射
SENSOR_TO_ID = {'Unknown': 0, 'TI_AWR1843': 1, 'TI_AWR_Series': 2}
NUM_SENSORS = len(SENSOR_TO_ID)

# 3. 平台状态映射
PLATFORM_TO_ID = {'Static': 0, 'Mobile': 1, 'Mixed': 2}
NUM_PLATFORMS = len(PLATFORM_TO_ID)

# 4. 数据类型映射
DATATYPE_TO_ID = {'RAD': 0, 'RAT': 1}
NUM_DATATYPES = len(DATATYPE_TO_ID)

# 5. 数值型特征的顺序
NUMERICAL_FEATURE_KEYS = [
    'azimuth_fov', 'max_range', 'max_velocity_or_NA', 
    'range_res', 'velocity_or_time_res', 'angular_res'
]
NUMERICAL_DIM = len(NUMERICAL_FEATURE_KEYS)

# 6. 元数据总表
METADATA_TABLE = {
    'CARRADA': {
        'sensor': SENSOR_TO_ID['Unknown'], 
        'platform': PLATFORM_TO_ID['Static'], 
        'datatype': DATATYPE_TO_ID['RAD'],
        'numerical': [90.0, 50.0, 13.43, 0.195, 0.42, 0.700]
    },
    # ... 其他数据集 ...
}