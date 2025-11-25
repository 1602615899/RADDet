import random
from torch.utils.data import Dataset
import numpy as np
import os, glob
import models.RADDet_finetune.loader as loader
import models.RADDet_finetune.helper as helper
from torchvision.transforms import ToTensor
import torch
from scipy.ndimage import zoom

def Create_RADDet_Finetune_Dataset_cart(config_file_name, valdatatype='validate'):
    config = loader.readConfig(config_file_name=config_file_name)
    config_data = config["DATA"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]    
    anchor_boxes_cart = loader.readAnchorBoxes(anchor_boxes_file="./models/RADDet_finetune/anchors_cartboxes.txt")
    
    train_dataset = RADDet_Finetune_cart(config_data=config_data,
                                        config_train=config_train,
                                        config_model=config_model,
                                        headoutput_shape=config_data["headoutput_shape"],
                                        anchors_cart=anchor_boxes_cart,
                                        cart_shape=config_data["cart_shape"],
                                        dType="train")
    validate_dataset = RADDet_Finetune_cart(config_data=config_data,
                                        config_train=config_train,
                                        config_model=config_model,
                                        headoutput_shape=config_data["headoutput_shape"],
                                        anchors_cart=anchor_boxes_cart,
                                        cart_shape=config_data["cart_shape"],
                                        dType=valdatatype)

    return train_dataset, validate_dataset

class RADDet_Finetune_cart(Dataset):
    def __init__(self, config_data, config_train, config_model, headoutput_shape,
                 transformer=ToTensor(), anchors_cart=None, cart_shape=None, dType="train", RADDir="RAD"):
        super(RADDet_Finetune_cart, self).__init__()
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
        self.cart_shape1 = [
            cart_shape[0],
            int(cart_shape[1] * 0.5),  # D
            int(cart_shape[2] * 0.5),  # H 
            int(cart_shape[3] * 0.5),  # W
            cart_shape[4]
        ]
        self.cart_shape2 = [
            cart_shape[0],
            int(cart_shape[1] * 2),    # D
            int(cart_shape[2] * 2),   # H
            int(cart_shape[3] * 2),   # W
            cart_shape[4]
        ]

        self.RADDir = RADDir
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes_cart = anchors_cart
        self.RAD_sequences_train = self.readSequences(mode="train")
        self.RAD_sequences_test = self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.RAD_sequences_train, self.RAD_sequences_validate = self.splitTrain(self.RAD_sequences_train)
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * len(self.RAD_sequences_train)) // self.batch_size
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        self.total_validate_batches = len(self.RAD_sequences_validate) // self.batch_size
        self.dtype = dType
        # self.transform = transformer
        self.transform = None
        self.scale_range = (0.7, 1.3)  # 3D缩放范围
        self.target_3d_shape = (256, 256, 64)  # [H, W, D]

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
            return self.trainDataCart(index)
        elif self.dtype == "validate":
            return self.validateDataCart(index)
        elif self.dtype == "test":
            return self.testDataCart(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def trainDataCart(self, index):
        """ Generate train data with batch size """
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")

        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_train[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
            RAD_data = helper.normalize_data(RAD_data)

            # RAD_data = helper.normalize_data(RAD_data)      # by zx, normalize data to [0, 1]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            # if random.random() < 0.5:
            #     RAD_data, transform_params = self.random_3d_augment(RAD_data)
            #     RAD_data = RAD_data.astype(np.float32)
            #     # current_anchors = self.get_scaled_anchors(transform_params['scale_factors'])
            #     # 调整边界框
            #     adjusted_boxes, adjusted_classes = self.adjust_3d_boxes(
            #         gt_instances["boxes"], 
            #         gt_instances["classes"],
            #         transform_params
            #     )
            #     # augtime += time.time() - start_time
            #     # print(f"增强时间: {augtime:.4f}s")
            # else:  # 50%概率不增强
            #     # current_anchors = self.anchor_boxes
            #     adjusted_boxes = gt_instances["boxes"]
            #     adjusted_classes = gt_instances["classes"]
            #     # augtime += time.time() - start_time
            #     # print(f"不增强时间: {augtime:.4f}s")
            
            # gt_instances["boxes"] = adjusted_boxes
            # gt_instances["classes"] = adjusted_classes  # 更新类别

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_list = self.encodeToCartBoxesLabels(gt_instances)
            gt_labels0, has_label0, raw_boxes0 = gt_list[0]
            gt_labels1, has_label1, raw_boxes1 = gt_list[1]
            gt_labels2, has_label2, raw_boxes2 = gt_list[2]
            index += 1
            gt_labels0 = np.stack(gt_labels0, axis=0)
            gt_labels1 = np.stack(gt_labels1, axis=0)
            gt_labels2 = np.stack(gt_labels2, axis=0)
            if has_label0:
                if self.transform:
                    return torch.tensor(RAD_data), torch.tensor(gt_labels0, dtype=torch.float32), torch.tensor(gt_labels1, dtype=torch.float32), \
                        torch.tensor(gt_labels2, dtype=torch.float32), torch.tensor(raw_boxes0, dtype=torch.float32)
                    # return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                    #     torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels2, gt_labels0, gt_labels1, raw_boxes0

    def testDataCart(self, index):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]

            RAD_data = helper.normalize_data(RAD_data)      # by zx, normalize data to [0, 1]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["test_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_list = self.encodeToCartBoxesLabels(gt_instances)
            gt_labels0, has_label0, raw_boxes0 = gt_list[0]
            gt_labels1, has_label1, raw_boxes1 = gt_list[1]
            gt_labels2, has_label2, raw_boxes2 = gt_list[2]
            index += 1
            gt_labels0 = np.stack(gt_labels0, axis=0)
            gt_labels1 = np.stack(gt_labels1, axis=0)
            gt_labels2 = np.stack(gt_labels2, axis=0)
            if has_label0:
                if self.transform:
                    return torch.tensor(RAD_data), torch.tensor(gt_labels0, dtype=torch.float32), torch.tensor(gt_labels1, dtype=torch.float32), \
                        torch.tensor(gt_labels2, dtype=torch.float32), torch.tensor(raw_boxes0, dtype=torch.float32)
                    # return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                    #     torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels2, gt_labels0, gt_labels1, raw_boxes0

    def validateDataCart(self, index):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_validate[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]

            RAD_data = helper.normalize_data(RAD_data)      # by zx, normalize data to [0, 1]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")
            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_list = self.encodeToCartBoxesLabels(gt_instances)
            gt_labels0, has_label0, raw_boxes0 = gt_list[0]
            gt_labels1, has_label1, raw_boxes1 = gt_list[1]
            gt_labels2, has_label2, raw_boxes2 = gt_list[2]
            index += 1
            gt_labels0 = np.stack(gt_labels0, axis=0)
            gt_labels1 = np.stack(gt_labels1, axis=0)
            gt_labels2 = np.stack(gt_labels2, axis=0)
            if has_label0:
                if self.transform:
                    return torch.tensor(RAD_data), torch.tensor(gt_labels0, dtype=torch.float32), torch.tensor(gt_labels1, dtype=torch.float32), \
                        torch.tensor(gt_labels2, dtype=torch.float32), torch.tensor(raw_boxes0, dtype=torch.float32)
                    # return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                    #     torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels2, gt_labels0, gt_labels1, raw_boxes0

    def encodeToCartBoxesLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xywh0 = np.zeros((self.config_data["max_boxes_per_frame"], 5))
        raw_boxes_xywh1 = np.zeros((self.config_data["max_boxes_per_frame"], 5))
        raw_boxes_xywh2 = np.zeros((self.config_data["max_boxes_per_frame"], 5))
        ### initialize gronud truth labels as np.zeros ###
        gt_labels0 = np.zeros(list(self.cart_shape[1:3]) + [len(self.anchor_boxes_cart)] +
                             [len(self.config_data["all_classes"]) + 5])
        gt_labels1 = np.zeros(list(self.cart_shape1[1:3]) + [len(self.anchor_boxes_cart)] +
                             [len(self.config_data["all_classes"]) + 5])
        gt_labels2 = np.zeros(list(self.cart_shape2[1:3]) + [len(self.anchor_boxes_cart)] +
                             [len(self.config_data["all_classes"]) + 5])

        has_label0 = False
        has_label1 = False
        has_label2 = False
        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xywh = gt_instances["cart_boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))

            # Process for gt_labels0
            grid_strid0 = self.cart_grid_strides[0]
            anchor_stage0 = self.anchor_boxes_cart
            box_xywh_scaled0 = box_xywh[np.newaxis, :].astype(np.float32)
            box_xywh_scaled0[:, :2] /= grid_strid0
            anchors_xywh0 = np.zeros([len(anchor_stage0), 4])
            anchors_xywh0[:, :2] = np.floor(box_xywh_scaled0[:, :2]) + 0.5
            anchors_xywh0[:, 2:] = anchor_stage0.astype(np.float32)

            iou_scaled0 = helper.iou2d(box_xywh_scaled0, anchors_xywh0)
            iou_mask0 = iou_scaled0 > 0.3

            if np.any(iou_mask0):
                xind, yind = np.floor(np.squeeze(box_xywh_scaled0)[:2]).astype(np.int32)
                if 0 <= xind < gt_labels0.shape[0] and 0 <= yind < gt_labels0.shape[1]:
                    gt_labels0[xind, yind, iou_mask0, 0:4] = box_xywh
                    gt_labels0[xind, yind, iou_mask0, 4:5] = 1.
                    gt_labels0[xind, yind, iou_mask0, 5:] = class_onehot
                    has_label0 = True
            if not np.any(iou_mask0):
                anchor_ind0 = np.argmax(iou_scaled0)
                xind, yind = np.floor(np.squeeze(box_xywh_scaled0)[:2]).astype(np.int32)
                if 0 <= xind < gt_labels0.shape[0] and 0 <= yind < gt_labels0.shape[1]:
                    gt_labels0[xind, yind, anchor_ind0, 0:4] = box_xywh
                    gt_labels0[xind, yind, anchor_ind0, 4:5] = 1.
                    gt_labels0[xind, yind, anchor_ind0, 5:] = class_onehot
                    has_label0 = True
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xywh0[i, :4] = box_xywh
                raw_boxes_xywh0[i, 4] = class_id

            # Process for gt_labels1
            grid_strid1 = self.cart_grid_strides[1]
            anchor_stage1 = self.anchor_boxes_cart
            box_xywh_scaled1 = box_xywh[np.newaxis, :].astype(np.float32)
            box_xywh_scaled1[:, :2] /= grid_strid1
            anchors_xywh1 = np.zeros([len(anchor_stage1), 4])
            anchors_xywh1[:, :2] = np.floor(box_xywh_scaled1[:, :2]) + 0.5
            anchors_xywh1[:, 2:] = anchor_stage1.astype(np.float32)

            iou_scaled1 = helper.iou2d(box_xywh_scaled1, anchors_xywh1)
            iou_mask1 = iou_scaled1 > 0.3

            if np.any(iou_mask1):
                xind, yind = np.floor(np.squeeze(box_xywh_scaled1)[:2]).astype(np.int32)
                if 0 <= xind < gt_labels1.shape[0] and 0 <= yind < gt_labels1.shape[1]:
                    gt_labels1[xind, yind, iou_mask1, 0:4] = box_xywh
                    gt_labels1[xind, yind, iou_mask1, 4:5] = 1.
                    gt_labels1[xind, yind, iou_mask1, 5:] = class_onehot
                    has_label1 = True
            if not np.any(iou_mask1):
                anchor_ind1 = np.argmax(iou_scaled1)
                xind, yind = np.floor(np.squeeze(box_xywh_scaled1)[:2]).astype(np.int32)
                if 0 <= xind < gt_labels1.shape[0] and 0 <= yind < gt_labels1.shape[1]:
                    gt_labels1[xind, yind, anchor_ind1, 0:4] = box_xywh
                    gt_labels1[xind, yind, anchor_ind1, 4:5] = 1.
                    gt_labels1[xind, yind, anchor_ind1, 5:] = class_onehot
                    has_label1 = True
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xywh1[i, :4] = box_xywh
                raw_boxes_xywh1[i, 4] = class_id

            # Process for gt_labels2
            grid_strid2 = self.cart_grid_strides[2]
            anchor_stage2 = self.anchor_boxes_cart
            box_xywh_scaled2 = box_xywh[np.newaxis, :].astype(np.float32)
            box_xywh_scaled2[:, :2] /= grid_strid2
            anchors_xywh2 = np.zeros([len(anchor_stage2), 4])
            anchors_xywh2[:, :2] = np.floor(box_xywh_scaled2[:, :2]) + 0.5
            anchors_xywh2[:, 2:] = anchor_stage2.astype(np.float32)

            iou_scaled2 = helper.iou2d(box_xywh_scaled2, anchors_xywh2)
            iou_mask2 = iou_scaled2 > 0.3

            if np.any(iou_mask2):
                xind, yind = np.floor(np.squeeze(box_xywh_scaled2)[:2]).astype(np.int32)
                if 0 <= xind < gt_labels2.shape[0] and 0 <= yind < gt_labels2.shape[1]:
                    gt_labels2[xind, yind, iou_mask2, 0:4] = box_xywh
                    gt_labels2[xind, yind, iou_mask2, 4:5] = 1.
                    gt_labels2[xind, yind, iou_mask2, 5:] = class_onehot
                    has_label2 = True
            if not np.any(iou_mask2):
                anchor_ind2 = np.argmax(iou_scaled2)
                xind, yind = np.floor(np.squeeze(box_xywh_scaled2)[:2]).astype(np.int32)
                if 0 <= xind < gt_labels2.shape[0] and 0 <= yind < gt_labels2.shape[1]:
                    gt_labels2[xind, yind, anchor_ind2, 0:4] = box_xywh
                    gt_labels2[xind, yind, anchor_ind2, 4:5] = 1.
                    gt_labels2[xind, yind, anchor_ind2, 5:] = class_onehot
                    has_label2 = True
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xywh2[i, :4] = box_xywh
                raw_boxes_xywh2[i, 4] = class_id

        gt_labels0 = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in [gt_labels0]]
        gt_labels1 = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in [gt_labels1]]
        gt_labels2 = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in [gt_labels2]]

        return [gt_labels0[0], has_label0, raw_boxes_xywh0], \
               [gt_labels1[0], has_label1, raw_boxes_xywh1], \
               [gt_labels2[0], has_label2, raw_boxes_xywh2]

    def getGridStrides(self):
        """ Get grid strides for 5D output shapes """
        strides0 = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        strides1 = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape1[1:4]))
        strides2 = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape2[1:4]))
        return np.array(strides0).astype(np.float32), np.array(strides1).astype(np.float32), np.array(strides2).astype(np.float32)

    def getCartGridStrides(self, ):
        """ Get grid strides """
        cart_output_shape = [int(self.config_model["input_shape"][0]), int(2 * self.config_model["input_shape"][0])]
        strides0 = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
        strides1 = (np.array(cart_output_shape) / np.array(self.cart_shape1[1:3]))
        strides2 = (np.array(cart_output_shape) / np.array(self.cart_shape2[1:3]))
        return np.array(strides0).astype(np.float32), np.array(strides1).astype(np.float32), np.array(strides2).astype(np.float32)

    def readSequences(self, mode):
        """ Read sequences from train/test directories. """
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(os.path.join(self.config_data["train_set_dir"], f"{self.RADDir}/*/*.npy"))
        elif mode == "test":
            sequences = glob.glob(os.path.join(self.config_data["test_set_dir"], f"{self.RADDir}/*/*.npy"))
        else:
            raise ValueError(f"{mode} type does not exist.")
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, \
                        Please double-check the data path or the data format.")
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
        scale_h, scale_w, scale_d = transform_params['scale_factors'] # (H_scale, W_scale, D_scale)

        # 裁剪的起始点和填充量
        start_h, start_w, start_d = transform_params['crop_starts']
        pad_h, pad_w, pad_d = transform_params['pads']
        
        target_h, target_w, target_d = self.target_3d_shape # (256, 256, 64)

        for box, cls in zip(boxes, classes):
            # box: [x_center/min, y_center/min, z_center/min, h_box, w_box, d_box]
            # 假设 x,y,z 是盒子中心坐标，h,w,d 是盒子尺寸
            # 如果是xmin,ymin,zmin，需要根据你的encodeToCartBoxesLabels函数逻辑来确定
            # 但从调整h,w,d的clip来看，似乎是xmin,ymin,zmin
            # 假设 box 是 [xmin, ymin, zmin, h, w, d] 形式
            xmin_orig, ymin_orig, zmin_orig, h_orig, w_orig, d_orig = box
            
            # 1. 缩放中心点和尺寸
            # 原始坐标和尺寸都是在 (H_orig, W_orig, D_orig) 空间下
            # xmin_orig 对应 H 轴
            # ymin_orig 对应 W 轴
            # zmin_orig 对应 D 轴

            scaled_xmin = xmin_orig * scale_h
            scaled_ymin = ymin_orig * scale_w
            scaled_zmin = zmin_orig * scale_d
            
            scaled_h = h_orig * scale_h
            scaled_w = w_orig * scale_w
            scaled_d = d_orig * scale_d

            # 2. 应用裁剪和填充的偏移
            # cropped region starts at (start_h, start_w, start_d)
            # padded region starts at (pad_h, pad_w, pad_d) inside the target_3d_shape
            
            # 新的 xmin 坐标 = 缩放后的原始坐标 - 裁剪的起始偏移 + 填充的起始偏移
            new_xmin = scaled_xmin - start_h + pad_h
            new_ymin = scaled_ymin - start_w + pad_w
            new_zmin = scaled_zmin - start_d + pad_d

            # 3. 计算新的xmax, ymax, zmax (在padded空间内)
            new_xmax = new_xmin + scaled_h
            new_ymax = new_ymin + scaled_w
            new_zmax = new_zmin + scaled_d

            # 4. 边界约束 (在 target_3d_shape 范围内)
            # 先对角点进行裁剪
            new_xmin = np.clip(new_xmin, 0, target_h) # xmin不能超过target_h
            new_ymin = np.clip(new_ymin, 0, target_w) # ymin不能超过target_w
            new_zmin = np.clip(new_zmin, 0, target_d) # zmin不能超过target_d

            new_xmax = np.clip(new_xmax, 0, target_h) # xmax不能超过target_h
            new_ymax = np.clip(new_ymax, 0, target_w) # ymax不能超过target_w
            new_zmax = np.clip(new_zmax, 0, target_d) # zmax不能超过target_d

            # 重新计算裁剪后的尺寸
            new_h = new_xmax - new_xmin
            new_w = new_ymax - new_ymin
            new_d = new_zmax - new_zmin
            
            # 5. 过滤条件
            # 确保尺寸有效 (至少1像素)
            if new_h < 1 or new_w < 1 or new_d < 1:
                continue

            # 新的体积
            adjusted_volume = new_h * new_w * new_d
            # 原始缩放后的体积
            original_scaled_volume = scaled_h * scaled_w * scaled_d 
            
            # 过滤条件
            # 保留至少30%缩放后体积的物体
            # 确保盒子仍然大部分可见且有效
            if (adjusted_volume >= original_scaled_volume * 0.3):
                adjusted_boxes.append([new_xmin, new_ymin, new_zmin, new_h, new_w, new_d])
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