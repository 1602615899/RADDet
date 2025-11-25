import webdataset as wds
import os
from tqdm import tqdm
import glob
import numpy as np
from io import BytesIO

import data_utils as du
import time

'''
all_datasets

'''
def decode_npy(sample):
    # time_start = time.time()
    # sample 是一个包含相关信息的字典
    byte_data = sample['input.npy']  # 获取字节数据
    # 使用 BytesIO 将字节数据转换为文件对象
    data = np.load(BytesIO(byte_data))  # 加载 .npy 文件
    # preprocess data
    data = du.preprocess_data(data)

    index = sample['__key__']  # 获取索引信息

    # print(f"load_data time: {time.time() - time_start:.3f}s")
    return index, data

'''
dataset = wds.WebDataset(["dataset-000.tar", "dataset-001.tar", "dataset-002.tar", "dataset-003.tar"])
dataset = wds.WebDataset("dataset-{000..003}.tar")

'''
def radar_datasets(data_type):
    root_dir = '/mnt/SrvUserDisk/ZhangXu/pretrain/rpt/datasets/data/'
    if data_type == 'RADDet':
        tar_name = root_dir + 'RADDet_{01..16}.tar'
        data_len = 8000     # train: 8126, test: 2032, all: 10158
    elif data_type == 'CARRADA':
        tar_name = root_dir + 'CARRADA_{01..03}.tar'
        data_len = 12666
    else:
        raise ValueError('Invalid data_type')
    
    # 创建 WebDataset 对象
    # shard 指几个tar文件，RADDet_{01..03}.tar
    dataset = (
        wds.WebDataset(tar_name, shardshuffle=True)
        .shuffle(1000)
        .map(decode_npy)
        # .batched(batch_size)
        .with_length(data_len)
        # .with_epoch(10)
    )

    return dataset