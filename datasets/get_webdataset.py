# import webdataset as wds
# import os
# from tqdm import tqdm
# import glob
# import numpy as np

# data_name = 'CARRADA'
# # data_name = 'RADDet'
# fold_idx = '01'

# if data_name == 'CARRADA':
#     data_dir = '/mnt/SrvDataDisk/Datasets_Radar/CARRADA'
#     sequences_all = glob.glob(os.path.join(data_dir, "Carrada_RAD", "*/*/*.npy"))
# elif data_name == 'RADDet':
#     data_dir = '/mnt/Disk/zx/data/RADDet_author'
#     sequences_train = glob.glob(os.path.join(data_dir, "train", "RAD/*/*.npy"))
#     sequences_test = glob.glob(os.path.join(data_dir, "test", "RAD/*/*.npy"))
#     sequences_all = sequences_train + sequences_test
# sequences_all.sort()

# index_start = 0
# index_end = 1999
# # 0-1000, 1000-2001, 2001-3001, 3001-4001, 4001-5001, 5001-6001, 6001-7001, 7001-8001, 8001-9001, 9001-:
# sequence = sequences_all[index_start:index_end]

# sink = wds.TarWriter(f"{data_name}_{fold_idx}.tar") # 使用TarWriter，准备将数据写入.tar文件

# index = index_start
# for filename in tqdm(sequence):
#     sample = np.load(filename)
#     data = {
#         "__key__": f"sample_{index:06d}",  # 当前样本的index
#         "input.npy": sample,  # 雷达数据
#     }
#     sink.write(data) # 将数据写入.tar文件
#     index += 1
# sink.close() # 关闭.tar文件

# print(f"Done! {index_end} to {index_start} samples saved in {data_name}_{fold_idx}.tar")

'''CARRADA
Done! 2000 samples saved in CARRADA_01.tar
Done! Samples from 1999 to 4000 saved in CARRADA_02.tar
Done! Samples from 4000 to 6000 saved in CARRADA_03.tar
'''

'''RADDet

'''


########################################################################################################
# 多线程
import webdataset as wds
import os
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置数据源
data_name = 'RADDet'
max_threads = 8  # 设置最大线程数
output_dir = '/mnt/SrvUserDisk/ZhangXu/pretrain/rpt/datasets/data/'  # 设置输出目录

# 设置数据目录路径和文件列表
if data_name == 'CARRADA':
    data_dir = '/mnt/SrvDataDisk/Datasets_Radar/CARRADA'
    sequences_all = glob.glob(os.path.join(data_dir, "Carrada_RAD", "*/*/*.npy"))
elif data_name == 'RADDet':
    data_dir = '/mnt/Disk/zx/data/RADDet_author'
    sequences_train = glob.glob(os.path.join(data_dir, "train", "RAD/*/*.npy"))
    sequences_test = glob.glob(os.path.join(data_dir, "test", "RAD/*/*.npy"))
    sequences_all = sequences_train + sequences_test
sequences_all.sort()

# 生成 index_list
index_list = []
gap = 500
for start in range(0, len(sequences_all), gap):
    end = min(start + gap, len(sequences_all))  # 确保不会超出范围
    index_list.append([start, end])

fold_idx = 1
for (index_start, index_end) in index_list:
    print(f"Processing tar {fold_idx:02d}...")
    # 设置索引范围
    if fold_idx == len(index_list):
        sequence = sequences_all[index_start:]
    else:
        sequence = sequences_all[index_start:index_end]

    # 打包文件路径
    output_tar = output_dir + f"{data_name}_{fold_idx:02d}.tar"

    # 定义将单个文件写入 .tar 的函数
    def process_and_write(filename, index):
        sample = np.load(filename)
        data = {
            "__key__": f"sample_{index:06d}",
            "input.npy": sample,
        }
        return data

    # 使用多线程写入 .tar 文件
    with wds.TarWriter(output_tar) as sink, ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_and_write, filename, idx): idx for idx, filename in enumerate(sequence, start=index_start)}
        
        # 处理进度条
        for future in tqdm(as_completed(futures), total=len(futures)):
            data = future.result()
            sink.write(data)

    print(f"Done! Samples from {index_start} to {index_end} saved in {output_tar}")

    fold_idx += 1
