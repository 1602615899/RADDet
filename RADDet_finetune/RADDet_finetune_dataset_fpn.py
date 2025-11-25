import random
import time
from torch.utils.data import Dataset
import numpy as np
import os, glob
import models.RADDet_finetune.loader as loader
import models.RADDet_finetune.helper as helper
from torchvision.transforms import ToTensor
import torch
from scipy.ndimage import zoom
import json


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

def Create_RADDet_Finetune_Dataset(config_file_name, valdatatype='test', sensor_config_path=None):
    config = loader.readConfig(config_file_name=config_file_name)
    config_data = config["DATA"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]    
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./models/RADDet_finetune/anchors.txt")

    # 加载传感器参数
    sensor_params = None
    if sensor_config_path and os.path.exists(sensor_config_path):
        with open(sensor_config_path, 'r') as f:
            sensor_configs = json.load(f)
            sensor_params = sensor_configs.get("RADDet", {})
    
    train_dataset = RADDet_Finetune(config_data, config_train, config_model,
                                 config_model["feature_out_shape"], anchor_boxes, dType="train", sensor_params=sensor_params)
    validate_dataset = RADDet_Finetune(config_data, config_train, config_model,
                                    config_model["feature_out_shape"], anchor_boxes, dType=valdatatype, sensor_params=sensor_params)

    return train_dataset, validate_dataset

class RADDet_Finetune(Dataset):
    def __init__(self, config_data, config_train, config_model, headoutput_shape,
                 anchors, transformer=ToTensor(), anchors_cart=None, cart_shape=None, dType="train", RADDir="RAD", sensor_params=None):
        super(RADDet_Finetune, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        
        self.headoutput_shape = headoutput_shape
        self.headoutput_shape1 = [
            headoutput_shape[0],
            int(headoutput_shape[1] * 0.5),  # D
            int(headoutput_shape[2] * 0.5),  # H 
            int(headoutput_shape[3] * 0.5),  # W
            headoutput_shape[4]
        ]
        self.headoutput_shape2 = [
            headoutput_shape[0],
            int(headoutput_shape[1] * 2),    # D
            int(headoutput_shape[2] * 2),   # H
            int(headoutput_shape[3] * 2),   # W
            headoutput_shape[4]
        ]
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RADDir = RADDir
        self.RAD_sequences_train = self.readSequences(mode="train")
        self.RAD_sequences_test = self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        # self.RAD_sequences_train, self.RAD_sequences_validate = self.splitTrain(self.RAD_sequences_train)
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * len(self.RAD_sequences_train)) // self.batch_size
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        # self.total_validate_batches = len(self.RAD_sequences_validate) // self.batch_size
        self.dtype = dType

        self.scale_range = (0.7, 1.3)  # 3D缩放范围
        self.target_3d_shape = (256, 256, 64)  # [H, W, D]
        # 统计值 - 与预训练保持一致
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
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            
            ### NOTE: Gloabl Normalization ###
            RAD_data = complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.mean_log) / self.std_log
            RAD_data = normalize_data(RAD_data)       

            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")
            
            if random.random() < 0:
                RAD_data, transform_params = self.random_3d_augment(RAD_data)
                RAD_data = RAD_data.astype(np.float32)
                adjusted_boxes, adjusted_classes = self.adjust_3d_boxes(
                    gt_instances["boxes"], 
                    gt_instances["classes"],
                    transform_params
                )
            else:  
                adjusted_boxes = gt_instances["boxes"]
                adjusted_classes = gt_instances["classes"]
                
            gt_instances["boxes"] = adjusted_boxes
            gt_instances["classes"] = adjusted_classes  # 更新类别

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_list = self.encodeToLabels(gt_instances)
            gt_labels0, has_label0, raw_boxes0 = gt_list[0]
            gt_labels1, has_label1, raw_boxes1 = gt_list[1]
            gt_labels2, has_label2, raw_boxes2 = gt_list[2]
            index += 1
            gt_labels0 = np.stack(gt_labels0, axis=0)
            gt_labels1 = np.stack(gt_labels1, axis=0)
            gt_labels2 = np.stack(gt_labels2, axis=0)
            if has_label0:
                return {
                    'data': torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),  # [1, H, W, D]
                    'label0': torch.tensor(gt_labels0, dtype=torch.float32),
                    'label1': torch.tensor(gt_labels1, dtype=torch.float32),
                    'label2': torch.tensor(gt_labels2, dtype=torch.float32),
                    'raw_boxes': torch.tensor(raw_boxes0, dtype=torch.float32),
                    'condition': self.condition_tensor, 
                    'raw_params': self.raw_params
                }

    def valData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_validate[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            # RAD_data = np.transpose(RAD_data, (3, 2, 0, 1))
            RAD_data = helper.normalize_data(RAD_data)      # by zx, normalize data to [0, 1]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_list = self.encodeToLabels(gt_instances)
            gt_labels0, has_label0, raw_boxes0 = gt_list[0]
            gt_labels1, has_label1, raw_boxes1 = gt_list[1]
            gt_labels2, has_label2, raw_boxes2 = gt_list[2]
            index += 1
            gt_labels0 = np.stack(gt_labels0, axis=0)
            gt_labels1 = np.stack(gt_labels1, axis=0)
            gt_labels2 = np.stack(gt_labels2, axis=0)
            if has_label0:
                return {
                    'data': torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),  # [1, H, W, D]
                    'label0': torch.tensor(gt_labels0, dtype=torch.float32),
                    'label1': torch.tensor(gt_labels1, dtype=torch.float32),
                    'label2': torch.tensor(gt_labels2, dtype=torch.float32),
                    'raw_boxes': torch.tensor(raw_boxes0, dtype=torch.float32),
                    'condition': self.condition_tensor, 
                    'raw_params': self.raw_params
                }

    def testData(self, index):
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = loader.readRAD(RAD_filename)
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
            gt_list = self.encodeToLabels(gt_instances)
            gt_labels0, has_label0, raw_boxes0 = gt_list[0]
            gt_labels1, _, _ = gt_list[1]
            gt_labels2, _, _ = gt_list[2]
            index += 1
            gt_labels0 = np.stack(gt_labels0, axis=0)
            gt_labels1 = np.stack(gt_labels1, axis=0)
            gt_labels2 = np.stack(gt_labels2, axis=0)
            if has_label0:
                return {
                    'data': torch.tensor(RAD_data[np.newaxis, :, :, :], dtype=torch.float32),  # [1, H, W, D]
                    'label0': torch.tensor(gt_labels0, dtype=torch.float32),
                    'label1': torch.tensor(gt_labels1, dtype=torch.float32),
                    'label2': torch.tensor(gt_labels2, dtype=torch.float32),
                    'raw_boxes': torch.tensor(raw_boxes0, dtype=torch.float32),
                    'condition': self.condition_tensor, 
                    'raw_params': self.raw_params
                }

    def getGridStrides(self):
        """ Get grid strides for 5D output shapes """
        strides0 = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        strides1 = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape1[1:4]))
        strides2 = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape2[1:4]))
        return np.array(strides0).astype(np.float32), np.array(strides1).astype(np.float32), np.array(strides2).astype(np.float32)

    def getCartGridStrides(self, ):
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
        raw_boxes_xyzwhd0 = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        raw_boxes_xyzwhd1 = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        raw_boxes_xyzwhd2 = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels0 = np.zeros(list(self.headoutput_shape[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])
        gt_labels1 = np.zeros(list(self.headoutput_shape1[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])
        gt_labels2 = np.zeros(list(self.headoutput_shape2[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])
        
        has_label0 = False
        has_label1 = False
        has_label2 = False

        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))
            
            # Process for gt_labels0
            grid_strid0 = self.grid_strides[0]
            anchor_stage0 = self.anchor_boxes
            box_xyzwhd_scaled0 = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled0[:, :3] /= grid_strid0
            anchorstage_xyzwhd0 = np.zeros([len(anchor_stage0), 6])
            anchorstage_xyzwhd0[:, :3] = np.floor(box_xyzwhd_scaled0[:, :3]) + 0.5
            anchorstage_xyzwhd0[:, 3:] = anchor_stage0.astype(np.float32)

            iou_scaled0 = helper.iou3d(box_xyzwhd_scaled0, anchorstage_xyzwhd0, self.input_size)
            iou_mask0 = iou_scaled0 > 0.3

            if np.any(iou_mask0):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled0)[:3]).astype(np.int32)
                if 0 <= xind < gt_labels0.shape[0] and 0 <= yind < gt_labels0.shape[1] and 0 <= zind < gt_labels0.shape[2]:
                    gt_labels0[xind, yind, zind, iou_mask0, 0:6] = box_xyzwhd
                    gt_labels0[xind, yind, zind, iou_mask0, 6:7] = 1.
                    gt_labels0[xind, yind, zind, iou_mask0, 7:] = class_onehot
                    has_label0 = True
            if not np.any(iou_mask0):
                anchor_ind0 = np.argmax(iou_scaled0)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled0)[:3]).astype(np.int32)
                if 0 <= xind < gt_labels0.shape[0] and 0 <= yind < gt_labels0.shape[1] and 0 <= zind < gt_labels0.shape[2]:
                    gt_labels0[xind, yind, zind, anchor_ind0, 0:6] = box_xyzwhd
                    gt_labels0[xind, yind, zind, anchor_ind0, 6:7] = 1.
                    gt_labels0[xind, yind, zind, anchor_ind0, 7:] = class_onehot
                    has_label0 = True
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd0[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd0[i, 6] = class_id
            
            # Process for gt_labels1
            grid_strid1 = self.grid_strides[1]
            anchor_stage1 = self.anchor_boxes
            box_xyzwhd_scaled1 = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled1[:, :3] /= grid_strid1
            anchorstage_xyzwhd1 = np.zeros([len(anchor_stage1), 6])
            anchorstage_xyzwhd1[:, :3] = np.floor(box_xyzwhd_scaled1[:, :3]) + 0.5
            anchorstage_xyzwhd1[:, 3:] = anchor_stage1.astype(np.float32)

            iou_scaled1 = helper.iou3d(box_xyzwhd_scaled1, anchorstage_xyzwhd1, self.input_size)
            iou_mask1 = iou_scaled1 > 0.3

            if np.any(iou_mask1):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled1)[:3]).astype(np.int32)
                if 0 <= xind < gt_labels1.shape[0] and 0 <= yind < gt_labels1.shape[1] and 0 <= zind < gt_labels1.shape[2]:
                    gt_labels1[xind, yind, zind, iou_mask1, 0:6] = box_xyzwhd
                    gt_labels1[xind, yind, zind, iou_mask1, 6:7] = 1.
                    gt_labels1[xind, yind, zind, iou_mask1, 7:] = class_onehot
                    has_label1 = True
            if not np.any(iou_mask1):
                anchor_ind1 = np.argmax(iou_scaled1)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled1)[:3]).astype(np.int32)
                if 0 <= xind < gt_labels1.shape[0] and 0 <= yind < gt_labels1.shape[1] and 0 <= zind < gt_labels1.shape[2]:
                    gt_labels1[xind, yind, zind, anchor_ind1, 0:6] = box_xyzwhd
                    gt_labels1[xind, yind, zind, anchor_ind1, 6:7] = 1.
                    gt_labels1[xind, yind, zind, anchor_ind1, 7:] = class_onehot
                    has_label1 = True
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd1[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd1[i, 6] = class_id

            # Process for gt_labels2
            grid_strid2 = self.grid_strides[2]
            anchor_stage2 = self.anchor_boxes
            box_xyzwhd_scaled2 = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled2[:, :3] /= grid_strid2
            anchorstage_xyzwhd2 = np.zeros([len(anchor_stage2), 6])
            anchorstage_xyzwhd2[:, :3] = np.floor(box_xyzwhd_scaled2[:, :3]) + 0.5
            anchorstage_xyzwhd2[:, 3:] = anchor_stage2.astype(np.float32)

            iou_scaled2 = helper.iou3d(box_xyzwhd_scaled2, anchorstage_xyzwhd2, self.input_size)
            iou_mask2 = iou_scaled2 > 0.3

            if np.any(iou_mask2):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled2)[:3]).astype(np.int32)
                if 0 <= xind < gt_labels2.shape[0] and 0 <= yind < gt_labels2.shape[1] and 0 <= zind < gt_labels2.shape[2]:
                    gt_labels2[xind, yind, zind, iou_mask2, 0:6] = box_xyzwhd
                    gt_labels2[xind, yind, zind, iou_mask2, 6:7] = 1.
                    gt_labels2[xind, yind, zind, iou_mask2, 7:] = class_onehot
                    has_label2 = True
            if not np.any(iou_mask2):
                anchor_ind2 = np.argmax(iou_scaled2)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled2)[:3]).astype(np.int32)
                if 0 <= xind < gt_labels2.shape[0] and 0 <= yind < gt_labels2.shape[1] and 0 <= zind < gt_labels2.shape[2]:
                    gt_labels2[xind, yind, zind, anchor_ind2, 0:6] = box_xyzwhd
                    gt_labels2[xind, yind, zind, anchor_ind2, 6:7] = 1.
                    gt_labels2[xind, yind, zind, anchor_ind2, 7:] = class_onehot
                    has_label2 = True
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd2[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd2[i, 6] = class_id

        gt_labels0 = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in [gt_labels0]]
        gt_labels1 = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in [gt_labels1]]
        gt_labels2 = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in [gt_labels2]]

        return [gt_labels0[0], has_label0, raw_boxes_xyzwhd0], \
               [gt_labels1[0], has_label1, raw_boxes_xyzwhd1], \
               [gt_labels2[0], has_label2, raw_boxes_xyzwhd2]

    def random_3d_augment(self, data):
        """三维数据增强核心方法"""
        # 输入数据形状: (H=256, W=256, D=64)
        h, w, d = data.shape
        
        # scale = random.uniform(*self.scale_range)
        scale_factors = (
            random.uniform(*self.scale_range),
            random.uniform(*self.scale_range),
            random.uniform(*self.scale_range)
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
        target_h, target_w, target_d, = self.target_3d_shape  # (64, 256, 256)
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

    def adjust_3d_boxes(self, boxes, classes, transform_params):
        """调整3D边界框坐标并同步过滤类别"""
        adjusted_boxes = []
        adjusted_classes = []
        scale_h, scale_w, scale_d = transform_params['scale_factors']

        for box, cls in zip(boxes, classes):
            x, y, z, h, w, d = box
            
            # ===== 修正缩放计算 =====
            scaled_h = h * scale_h
            scaled_w = w * scale_w
            scaled_d = d * scale_d
            
            # 应用裁剪偏移
            start_h, start_w, start_d = transform_params['crop_starts']
            pad_h, pad_w, pad_d = transform_params['pads']
            new_x = (x * scale_h - (start_h - pad_h))
            new_y = (y * scale_w - (start_w - pad_w))
            new_z = (z * scale_d - (start_d - pad_d))

            new_h = scaled_h
            new_w = scaled_w
            new_d = scaled_d
            
            # 边界约束
            new_x = np.clip(new_x, 0, self.target_3d_shape[0]-1)  # H方向
            new_y = np.clip(new_y, 0, self.target_3d_shape[1]-1)  # W方向
            new_z = np.clip(new_z, 0, self.target_3d_shape[2]-1)  # D方向
            
            new_h = np.clip(new_h, 1, self.target_3d_shape[0] - new_y)
            new_w = np.clip(new_w, 1, self.target_3d_shape[1] - new_x)
            new_d = np.clip(new_d, 1, self.target_3d_shape[2] - new_z)
            
            original_volume = w * h * d
            scaled_volume = scaled_w * scaled_h * scaled_d
            adjusted_volume = new_w * new_h * new_d

            if (
                adjusted_volume >= scaled_volume * 0.3 and  # 保留至少30%缩放后体积
                (new_x + new_w) <= self.target_3d_shape[0] and  # 完全在有效区域内
                (new_y + new_h) <= self.target_3d_shape[1] and
                (new_z + new_d) <= self.target_3d_shape[2] and
                (adjusted_volume / original_volume) > 0.1  # 相对原始体积保留至少10%
            ):
                adjusted_boxes.append([new_x, new_y, new_z, new_w, new_h, new_d])
                adjusted_classes.append(cls)
        
        return np.array(adjusted_boxes), adjusted_classes

    def get_scaled_anchors(self, scale_factors):
        """根据3D缩放因子动态调整anchor尺寸"""
        # scale_factors格式：(depth_scale, height_scale, width_scale)
        # anchors原始格式：[W, H, D]（需根据实际数据维度顺序调整）
        scaled_anchors = self.anchor_boxes.copy()
        
        # 维度对应关系：scale_factors -> (D, H, W) 
        # anchor维度 -> (W, H, D) （根据实际数据格式调整）
        scaling = np.array([scale_factors[2],  # scale_w for anchor W (index 0)
                        scale_factors[1],  # scale_h for anchor H (index 1)
                        scale_factors[0]]) # scale_d for anchor D (index 2)
        # 仅调整anchor尺寸（whd），保持位置不变
        
        scaled_anchors = scaled_anchors * scaling 
        
        return scaled_anchors
    


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