from torch.utils.data import Dataset
import numpy as np
import os, glob
import models.RADDet_finetune.loader as loader
import models.RADDet_finetune.helper as helper
from torchvision.transforms import ToTensor
import torch
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob, os
import json
from scipy.ndimage import zoom
import time
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob, os
import json
from scipy.ndimage import zoom
import time
from tqdm import tqdm

class RADDet_Finetune(Dataset):
    def __init__(self, config_data, config_train, config_model, headoutput_shape,
                 anchors, transformer=None, anchors_cart=None, cart_shape=None, dType="train", RADDir="RAD"):
        super(RADDet_Finetune, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RADDir = RADDir
        self.RAD_sequences_train = self.readSequences(mode="train")
        self.RAD_sequences_test = self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.RAD_sequences_train, self.RAD_sequences_validate = self.splitTrain(self.RAD_sequences_train)
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * len(self.RAD_sequences_train)) // self.batch_size
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        self.total_validate_batches = len(self.RAD_sequences_validate) // self.batch_size
        self.dtype = dType
        self.transform = transformer
        
        # RADDet数据集参数 - 与预训练保持一致
        self.dataset_params = {
            'range_res': float(0.1953125),      # 距离分辨率
            'angle_res': float(np.deg2rad(0.67)),  # 角度分辨率 (约0.67度)
            'vel_res': float(0.41968),          # 速度分辨率
            'time_res': float(0.05),            # 时间分辨率
            'has_velocity': bool(True),         # 有速度维度
            'dataset_id': int(1)                # RADDet标识
        }
        
        # 统计值 - 与预训练保持一致
        self.mean_log = 3.243838
        self.std_log = 0.643712

    def __len__(self):
        if self.dtype == "train":
            return len(self.RAD_sequences_train)
        elif self.dtype == "validate":
            return len(self.RAD_sequences_validate)
        elif self.dtype == "test":
            return len(self.RAD_sequences_test)
        else:
            raise ValueError("This type of dataset does not exist.")

    def __getitem__(self, index):
        if self.dtype == "train":
            return self.trainData(index)
        elif self.dtype == "validate":
            return self.valData(index)
        elif self.dtype == "test":
            return self.testData(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def trainData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_train[index]
            RAD_complex = readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            
            ### NOTE: Gloabl Normalization ###
            RAD_data = complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.mean_log) / self.std_log
            RAD_data = normalize_data(RAD_data)    
            
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            gt_labels = np.stack(gt_labels, axis=0)
            index += 1
            if has_label:
                # 返回与预训练一致的字典格式
                return {
                    'data': torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),  # [1, H, W, D]
                    'label': torch.tensor(gt_labels, dtype=torch.float32),
                    'raw_boxes': torch.tensor(raw_boxes, dtype=torch.float32),
                    'dataset_id': 1,  # RADDet标识
                    'dataset_params': self.dataset_params,
                    'dataset_name': 'RADDet'
                }

    def valData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_validate[index]
            RAD_complex = readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            
            ### NOTE: Gloabl Normalization ###
            RAD_data = complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.mean_log) / self.std_log
            RAD_data = normalize_data(RAD_data)    
            
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            gt_labels = np.stack(gt_labels, axis=0)
            index += 1
            if has_label:
                # 返回与预训练一致的字典格式
                return {
                    'data': torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),  # [1, H, W, D]
                    'label': torch.tensor(gt_labels, dtype=torch.float32),
                    'raw_boxes': torch.tensor(raw_boxes, dtype=torch.float32),
                    'dataset_id': 1,  # RADDet标识
                    'dataset_params': self.dataset_params,
                    'dataset_name': 'RADDet'
                }

    def testData(self, index):
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            
            ### NOTE: Gloabl Normalization ###
            RAD_data = complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.mean_log) / self.std_log
            RAD_data = normalize_data(RAD_data)    

            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["test_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            gt_labels = np.stack(gt_labels, axis=0)
            index += 1
            if has_label:
                # 返回与预训练一致的字典格式
                return {
                    'data': torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),  # [1, H, W, D]
                    'label': torch.tensor(gt_labels, dtype=torch.float32),
                    'raw_boxes': torch.tensor(raw_boxes, dtype=torch.float32),
                    'dataset_id': 1,  # RADDet标识
                    'dataset_params': self.dataset_params,
                    'dataset_name': 'RADDet'
                }

    def getGridStrides(self):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    def getCartGridStrides(self):
        """ Get grid strides """
        if self.cart_shape is not None:
            cart_output_shape = [int(self.config_model["input_shape"][0]),
                                 int(2 * self.config_model["input_shape"][0])]
            strides = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
            return np.array(strides).astype(np.float32)
        else:
            return None

    def readSequences(self, mode):
        """ Read sequences from train/test directories. """
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(os.path.join(self.config_data["train_set_dir"], f"{self.RADDir}/*/*.npy"))
        else:
            sequences = glob.glob(os.path.join(self.config_data["test_set_dir"], f"{self.RADDir}/*/*.npy"))
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, "
                             "Please double-check the data path or the data format.")
        return sequences

    def splitTrain(self, train_sequences):
        """ Split train set to train and validate """
        total_num = len(train_sequences)
        validate_num = int(0.1 * total_num)
        if self.config_train["if_validate"]:
            return train_sequences[:total_num-validate_num], \
                train_sequences[total_num-validate_num:]
        else:
            return train_sequences, train_sequences[total_num-validate_num:]

    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])

        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))

            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = helper.iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                      self.input_size)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
        gt_labels = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels]
        return gt_labels, has_label, raw_boxes_xyzwhd

def Create_RADDet_Finetune_Dataset(config_file_name, valdatatype='test'):
    config = loader.readConfig(config_file_name=config_file_name)
    config_data = config["DATA"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]    
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./models/RADDet_finetune/anchors.txt")
    
    train_dataset = RADDet_Finetune(config_data, config_train, config_model,
                                 config_model["feature_out_shape"], anchor_boxes, dType="train")
    validate_dataset = RADDet_Finetune(config_data, config_train, config_model,
                                    config_model["feature_out_shape"], anchor_boxes, dType=valdatatype)

    return train_dataset, validate_dataset

####################################################################################################
# 工具函数 - 与预训练保持一致

def normalize_data(data):
    '''
    normalize data to [0, 1]
    '''
    data_min = data.min()
    data_max = data.max()
    
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
    output_array = getLog(output_array, scalar=1., log_10=True)
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