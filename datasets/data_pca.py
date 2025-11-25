import os
import glob
import numpy as np
from tqdm import tqdm
import data_utils as du
from sklearn.decomposition import PCA

from concurrent.futures import ThreadPoolExecutor

# data_type = 'CARRADA'  
data_type = 'RADDet'
RADDet_ADC = True
if data_type == 'CARRADA':
    RAD_data_dir = '/mnt/SrvDataDisk/Datasets_Radar/CARRADA'
    RAD_sequences_all = glob.glob(os.path.join(RAD_data_dir, "Carrada_RAD", "*/*/*.npy"))

    mean_log = 3.214459
    std_log = 0.644378
    out_dir = 'CARRADA/Carrada_RAD_pca_32'
    num = -41

elif data_type == 'RADDet':
    RAD_data_dir = '/home/ljm/workspace/RADDet_author'
    RAD_sequences_train = glob.glob(os.path.join(RAD_data_dir, "train", "RAD/*/*.npy"))
    RAD_sequences_test = glob.glob(os.path.join(RAD_data_dir, "test", "RAD/*/*.npy"))
    RAD_sequences_all = RAD_sequences_test

    mean_log = 3.243985
    std_log = 0.643747
    out_dir = '/mnt/Disk/zx/data/RADDet_pca_32/test/RDA'
    num = -18

# 这里叫RDA，因为是以RDA格式存储的，读取时直接reshape(-1,A_target_channel)做IPCA，再调整到需要的形状

RAD_sequences_all.sort()
print(f'{len(RAD_sequences_all)} sequences found.')



num_workers = 4
A_target_channel = 64

def process_file(filename):
    RAD_data = du.readRAD(filename).transpose(0, 2, 1)      # RDA
    RAD_data = du.preprocess_data(RAD_data)
    RAD_data = (RAD_data - mean_log) / std_log
    RAD_data = du.normalize_data(RAD_data)

    # 将数据展开为 2D 矩阵以便应用 PCA
    data_2d = RAD_data.reshape(-1, RAD_data.shape[2])  # shape = (256*64, 256)

    pca = PCA(n_components=A_target_channel)
    reduced_data = pca.fit_transform(data_2d)  # shape = (256*64, 8)

    # 将降维后的数据恢复为 [256, 64, 8] 进行保存
    reduced_data_3d = reduced_data.reshape(256, 64, A_target_channel)

    # reduced_data_3d = reduced_data_3d.transpose(0, 2, 1)    # RAD

    # 保存降维后的数据
    out_filename = out_dir + filename[num:]
    # if not os.path.exists(out_filename[:-10]):
    #     os.makedirs(out_filename[:-10])
    np.save(out_filename, reduced_data_3d)
    # np.save(f'./pca_data/test{RAD_sequences_all.index(filename)}.npy', reduced_data_3d)

# 使用 ThreadPoolExecutor 进行多线程处理
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    list(tqdm(executor.map(process_file, RAD_sequences_all), total=len(RAD_sequences_all), desc="Processing sequences"))


'''
####################### 恢复数据 #######################
reduced_data_2d = reduced_data_3d.reshape(-1, A_target_channel)
# 使用 PCA 的逆变换将数据从 8 维恢复到 256 维
reconstructed_data_2d = pca.inverse_transform(reduced_data_2d)  # shape = (256*64, 256)
reconstructed_data_3d = reconstructed_data_2d.reshape(256, 64, 256)

# 再归一化一次会降低精度，但也没降太多，A_target_channel越大，影响越小，要再归一化吗
reconstructed_data_3d = du.normalize_data(reconstructed_data_3d)
'''



# for index in tqdm(range(len(RAD_sequences_all)), desc="Processing sequences"):
#     # if index > 1:
#     #     break

#     filename = RAD_sequences_all[index]
#     RAD_data = du.readRAD(filename).transpose(0, 2, 1)
#     RAD_data = du.preprocess_data(RAD_data)
#     RAD_data = (RAD_data - mean_log) / std_log
#     RAD_data = du.normalize_data(RAD_data)

#     # 将数据展开为 2D 矩阵以便应用 PCA
#     data_2d = RAD_data.reshape(-1, RAD_data.shape[2])  # shape = (256*64, 256)

#     # 使用 PCA 将 256 维降到 A_target_channel 维
#     A_target_channel = 64

#     pca = PCA(n_components=A_target_channel)
#     reduced_data = pca.fit_transform(data_2d)  # shape = (256*64, 8)
#     # 将降维后的数据恢复为 [256, 64, 8] 进行保存
#     reduced_data_3d = reduced_data.reshape(256, 64, A_target_channel)

#     # 保存降维后的数据
#     # np.save(out_dir + filename[num:], reduced_data_3d)
#     np.save(f'./pca_data/test{index}.npy', reduced_data_3d)

    