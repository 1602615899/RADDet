import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from models.RADDet_finetune.yolo_loss import RadDetLoss
import models.RADDet_finetune.loader as loader

from utils import layernorm, layernorm_3d

def iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Numpy version of 3D bounding box IOU calculation
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou

def nms(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes


def nmsOverClass(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        best_bboxes = []
        ### NOTE: start looping over boxes to find the best one ###
        while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 6])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = iou3d(best_bbox[np.newaxis, :6], bboxes[:, :6], \
                        input_size)
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            bboxes[:, 6] = bboxes[:, 6] * weight
            score_mask = bboxes[:, 6] > 0.
            bboxes = bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes


def yoloheadToPredictions(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 8], where 8 means
    [x, y, z, w, h, d, score, class_index]"""
    prediction = yolohead_output.reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 7:], axis=-1)
    predictions = np.concatenate([prediction[:, :7], \
                                  np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 6] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions


def decodeYolo(input, input_size, anchor_boxes, scale):
    x = rearrange(input, "n c d h w -> n h w d c")
    grid_size = x.shape[1:4]
    grid_strides = input_size / torch.tensor(list(grid_size), device=input.device)  # 16, 16, 16
    g0, g1, g2 = grid_size
    pred_raw = rearrange(x, "b h w a (c c1)-> b h w a c c1", c1=13)   # (None, 16, 16, 4, 78) ---> (None, 16, 16, 4, 6, 13)
    raw_xyz = pred_raw[:, :, :, :, :, :3]
    raw_whd = pred_raw[:, :, :, :, :, 3:6]
    raw_conf = pred_raw[:, :, :, :, :, 6:7]
    raw_prob = pred_raw[:, :, :, :, :, 7:]

    xx, yy, zz = torch.meshgrid([torch.arange(0, g0), torch.arange(0, g1), torch.arange(0, g2)])
    # xx = rearrange(xx, "h w c -> w h c")
    # yy = rearrange(yy, "h w c -> w h c")
    # zz = rearrange(zz, "h w c -> w h c")
    xyz_grid = [yy, xx, zz]
    xyz_grid = torch.stack(xyz_grid, dim=-1).to(input.device)
    xyz_grid = torch.unsqueeze(xyz_grid, 3)
    xyz_grid = rearrange(xyz_grid, "h w c a d -> w h c a d")
    xyz_grid = torch.unsqueeze(xyz_grid, 0)
    xyz_grid = torch.tile(xyz_grid, (input.shape[0], 1, 1, 1, len(anchor_boxes), 1))
    xyz_grid = xyz_grid.to(torch.float32)

    ### NOTE: not sure about this SCALE, but it appears in YOLOv4 tf version ###
    pred_xyz = ((torch.sigmoid(raw_xyz) * scale) - 0.5 * (scale - 1) + xyz_grid) * grid_strides
    ###---------------- clipping values --------------------###
    raw_whd = torch.clamp(raw_whd, 1e-12, 1e12)
    ###-----------------------------------------------------###
    pred_whd = torch.exp(raw_whd) * anchor_boxes
    pred_xyzwhd = torch.cat((pred_xyz, pred_whd), dim=-1)

    pred_conf = torch.sigmoid(raw_conf)
    pred_prob = torch.sigmoid(raw_prob)

    results = torch.cat((pred_xyzwhd, pred_conf, pred_prob), dim=-1)
    return pred_raw, results

def decodeYolo_fpn(input, input_size, anchor_boxes, scale):
    x = rearrange(input, "n c h w d -> n h w d c")
    grid_size = x.shape[1:4]
    grid_strides = input_size / torch.tensor(list(grid_size), device=input.device)  # 16, 16, 16
    scaled_anchors  = anchor_boxes / grid_strides
    g0, g1, g2 = grid_size
    pred_raw = rearrange(x, "b h w a (c c1)-> b h w a c c1", c1=13)   # (None, 16, 16, 4, 78) ---> (None, 16, 16, 4, 6, 13)
    raw_xyz = pred_raw[:, :, :, :, :, :3]
    raw_whd = pred_raw[:, :, :, :, :, 3:6]
    raw_conf = pred_raw[:, :, :, :, :, 6:7]
    raw_prob = pred_raw[:, :, :, :, :, 7:]

    xx, yy, zz = torch.meshgrid([torch.arange(0, g0), torch.arange(0, g1), torch.arange(0, g2)])
    # xx = rearrange(xx, "h w c -> w h c")
    # yy = rearrange(yy, "h w c -> w h c")
    # zz = rearrange(zz, "h w c -> w h c")
    xyz_grid = [yy, xx, zz]
    xyz_grid = torch.stack(xyz_grid, dim=-1).to(input.device)
    xyz_grid = torch.unsqueeze(xyz_grid, 3)
    xyz_grid = rearrange(xyz_grid, "h w c a d -> w h c a d")
    xyz_grid = torch.unsqueeze(xyz_grid, 0)
    xyz_grid = torch.tile(xyz_grid, (input.shape[0], 1, 1, 1, len(anchor_boxes), 1))
    xyz_grid = xyz_grid.to(torch.float32)

    ### NOTE: not sure about this SCALE, but it appears in YOLOv4 tf version ###
    pred_xyz = ((torch.sigmoid(raw_xyz) * scale) - 0.5 * (scale - 1) + xyz_grid) * grid_strides
    ###---------------- clipping values --------------------###
    raw_whd = torch.clamp(raw_whd, 1e-12, 1e12)
    ###-----------------------------------------------------###
    pred_whd = torch.exp(raw_whd) * scaled_anchors
    pred_xyzwhd = torch.cat((pred_xyz, pred_whd), dim=-1)

    pred_conf = torch.sigmoid(raw_conf)
    pred_prob = torch.sigmoid(raw_prob)

    results = torch.cat((pred_xyzwhd, pred_conf, pred_prob), dim=-1)
    return pred_raw, results


def decodeYolo_cart(x, anchor_boxes):
        output_size = [256, 512]
        strides = torch.tensor(output_size) / torch.tensor(list(x.shape[1:3]))
        strides = strides.to(x.device)
        raw_xy, raw_wh, raw_conf, raw_prob = x[..., 0:2], x[..., 2:4], x[..., 4:5], x[..., 5:]
        xx, yy = torch.meshgrid([torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2])])
        xy_grid = [xx.T, yy.T]
        xy_grid = torch.unsqueeze(torch.stack(xy_grid, dim=-1), dim=-2).to(x.device)
        xy_grid = torch.unsqueeze(rearrange(xy_grid, "b c h w -> c b h w"), dim=0)
        xy_grid = torch.tile(xy_grid, (x.shape[0], 1, 1, 6, 1)).to(torch.float32)
        scale = 1
        pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides
        ###---------------- clipping values --------------------###
        raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        # pred_wh = torch.exp(raw_wh) * self.anchor_boxes
        pred_wh = torch.exp(raw_wh) * torch.tensor(anchor_boxes).to(raw_wh.device) # by zx
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)
        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

def extractYoloInfo(yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category


def losss(pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = extractYoloInfo(gt)

        box_loss = gt_conf * (torch.square(pred_box[..., :2] - gt_box[..., :2]) +
                              torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))
        iou = tf_iou2d(torch.unsqueeze(pred_box, dim=-2), raw_boxes[:, None, None, None, :, :])
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)
        gt_conf_negative = (1.0 - gt_conf) * (max_iou < 0.3).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2)
        alpha = 0.01
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        conf_loss = conf_focal * (gt_conf * bce(target=gt_conf, input=raw_conf)
                                  + alpha * gt_conf_negative * bce(target=gt_conf, input=raw_conf))
        ### NOTE: category loss function ###
        category_loss = gt_conf * bce(target=gt_category, input=raw_category)

        ### NOTE: combine together ###
        box_loss_all = torch.mean(torch.sum(box_loss, dim=[1, 2, 3, 4]))
        box_loss_all *= 1e-1
        conf_loss_all = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        
        category_loss_all = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
        
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all


def tf_iou2d(box_xywh_1, box_xywh_2):
    """ Tensorflow version of 3D bounding box IOU calculation
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = torch.maximum(bottom_right - left_top,
                                 torch.zeros(bottom_right.shape, dtype=bottom_right.dtype, device=box_xywh_1.device))
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = torch.nan_to_num(torch.div(intersection_area, union_area + 1e-10), 0.0)
    return iou

class singleLayerHead(nn.Module):
    def __init__(self, num_anchors, num_class, last_channel, in_feature_size):
        super(singleLayerHead, self).__init__()
        self.num_anchor = num_anchors
        self.num_class = num_class
        self.last_channel = last_channel
        self.in_feature_size = in_feature_size

        final_output_channels = int(last_channel * self.num_anchor * (num_class + 7))  # 312
        self.final_output_reshape = [-1] + [int(last_channel),
                                            int(self.num_anchor) * (num_class + 7)] + list(in_feature_size[2:])

        self.conv1 = nn.Conv2d(in_channels=self.in_feature_size[1],
                               out_channels=self.in_feature_size[1]*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True,
                               )
        # self.bn1 = nn.BatchNorm2d(self.in_feature_size[1]*2)
        self.bn1 = layernorm(self.in_feature_size[1]*2)
        # self.bn1 = nn.LayerNorm([self.in_feature_size[1]*2, in_feature_size[2], in_feature_size[3]])
        
        self.conv2 = nn.Conv2d(in_channels=self.in_feature_size[1]*2,
                               out_channels=final_output_channels,
                               kernel_size=1,
                               stride=1,
                               bias=True
                               )

        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.GELU()
        self.relu = nn.LeakyReLU()
        # self.relu = nn.Identity()
        
        # self.init_weights()  # 初始化权重

    def forward(self, input_feature):
        x = self.relu(self.bn1(self.conv1(input_feature)))
        x = self.conv2(x)
        x = x.view(self.final_output_reshape)
        return x

    def init_weights(self):
        # 对 conv1 和 conv2 的权重初始化
        # Conv1 使用 Kaiming 初始化是因为它连接了 ReLU 激活，专门设计为适配非线性函数。
        # Conv2 使用 Xavier 初始化是因为它是输出层，无非线性激活，更适合 Xavier 初始化保持梯度平衡。
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        
class singleLayerHead3D(nn.Module):
    def __init__(self, num_anchors, num_class, last_channel, channel):
        super(singleLayerHead3D, self).__init__()
        self.num_anchor = num_anchors
        self.num_class = num_class
        self.channel = channel

        final_output_channels = int(self.num_anchor * (num_class + 7))  # 78
        # self.final_output_reshape = [-1] + [int(self.num_anchor) * (num_class + 7)] + list(in_feature_size[1:])

        self.conv1 = nn.Conv3d(in_channels=self.channel,
                               out_channels=self.channel*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True,
                               )
        self.bn1 = nn.BatchNorm3d(self.channel*2)

        self.conv2 = nn.Conv3d(in_channels=self.channel*2,
                               out_channels=final_output_channels,
                               kernel_size=1,
                               stride=1,
                               bias=True
                               )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_feature):
        x = self.relu(self.bn1(self.conv1(input_feature)))
        x = self.conv2(x)
        return x

class singleLayerHead_v2(nn.Module):
    def __init__(self, num_anchors, num_class, last_channel, in_feature_size):
        super(singleLayerHead_v2, self).__init__()
        self.num_anchor = num_anchors
        self.num_class = num_class
        self.last_channel = last_channel
        self.in_feature_size = in_feature_size

        final_output_channels = int(last_channel * self.num_anchor * (num_class + 7))  # 312
        self.final_output_reshape = [-1] + [int(last_channel),
                                            int(self.num_anchor) * (num_class + 7)] + list(in_feature_size[2:])

        self.conv1 = nn.Conv2d(in_channels=self.in_feature_size[1],
                               out_channels=self.in_feature_size[1]*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True,
                               )
        # self.bn1 = nn.BatchNorm2d(self.in_feature_size[1]*2)
        self.bn1 = layernorm(self.in_feature_size[1]*2)
        # self.bn1 = nn.LayerNorm([self.in_feature_size[1]*2, in_feature_size[2], in_feature_size[3]])
        
        self.conv2 = nn.Conv2d(in_channels=self.in_feature_size[1]*2,
                               out_channels=final_output_channels,
                               kernel_size=1,
                               stride=1,
                               bias=True
                               )

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.GELU()
        # self.relu = nn.LeakyReLU()
        # self.relu = nn.Identity()
        
        # self.init_weights()  # 初始化权重

    def forward(self, input_feature):
        x = self.relu(self.bn1(self.conv1(input_feature)))
        x = self.conv2(x)
        x = x.view(self.final_output_reshape)
        return x

    def init_weights(self):
        # 对 conv1 和 conv2 的权重初始化
        # Conv1 使用 Kaiming 初始化是因为它连接了 ReLU 激活，专门设计为适配非线性函数。
        # Conv2 使用 Xavier 初始化是因为它是输出层，无非线性激活，更适合 Xavier 初始化保持梯度平衡。
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)


import torch.nn.functional as F
class singleLayerHead_zx(nn.Module):
    def __init__(self, num_anchors, num_class, last_channel, in_feature_size,
                 embed_dim, input_size
                 ):
        super(singleLayerHead_zx, self).__init__()
        self.num_anchor = num_anchors
        self.num_class = num_class
        self.last_channel = last_channel
        self.in_feature_size = in_feature_size

        final_output_channels = int(last_channel * self.num_anchor * (num_class + 7))  # 312
        self.final_output_reshape = [-1] + [int(last_channel),
                                            int(self.num_anchor) * (num_class + 7)] + list(in_feature_size[2:])
        
        self.conv1x1 = nn.Conv2d(embed_dim*input_size[0], final_output_channels, kernel_size=1, stride=1, bias=True)

        self.ln1 = layernorm(final_output_channels)

        self.linear = nn.Linear(final_output_channels, final_output_channels)

        if input_size[1] == 16:
            self.maxpooling = nn.Identity()
        elif input_size[1] == 32:
            self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif input_size[1] == 64:
            self.maxpooling = nn.MaxPool2d(kernel_size=4, stride=4)
        else:
            raise ValueError("input_size should be 16, 32, or 64")

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.GELU()
        # self.relu = nn.LeakyReLU()
        # self.relu = nn.Identity()
        
        # self.init_weights()  # 初始化权重

    def forward(self, x):
        '''
        input: [batch_size, embed_dim, gridsize[0], [1], [2]]
        '''
        B, Dim, grid_D, grid_R, grid_A = x.shape
        x = x.reshape(B, Dim*grid_D, grid_R, grid_A)
        # 1x1 conv
        x = self.conv1x1(x)     # [B, C, gridsize[1], gridsize[2]] == [B, 312, 32, 32]

        # interpolate or Maxpooling
        x = self.maxpooling(x)  # [B, C, gridsize[1]/2, gridsize[2]/2] == [B, 312, 16, 16]
        # x = F.interpolate(x, size=self.in_feature_size[-2:], mode='bilinear', align_corners=True)  # [B, C, 256, 256]

        # layer norm
        x = self.ln1(x)

        # linear
        x = x + self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = x.view(self.final_output_reshape)

        return x

    def init_weights(self):
        # 对 conv1 和 conv2 的权重初始化
        # Conv1 使用 Kaiming 初始化是因为它连接了 ReLU 激活，专门设计为适配非线性函数。
        # Conv2 使用 Xavier 初始化是因为它是输出层，无非线性激活，更适合 Xavier 初始化保持梯度平衡。
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)


if __name__ == "__main__":
    input_features = torch.randn(3, 256, 16, 16)
    config = loader.readConfig(config_file_name="/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Dujialun/RADDet_Pytorch-main/config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]

    anchors_layer = anchor_boxes = loader.readAnchorBoxes(
        anchor_boxes_file="/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Dujialun/RADDet_Pytorch-main/anchors.txt")  # load anchor boxes with order
    num_classes = len(config_data["all_classes"])

    input_size = list(config_model["input_shape"])
    input_channels = input_size[-1]
    num_class = len(config_data["all_classes"])
    yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
    focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]

    # yolo_head = yoloHead(input_features, anchor_boxes, num_class)
    # output = yolo_head(input_features)
    input_tensor = torch.randn(3, 4, 78, 16, 16)
    pred_raw, pred = decodeYolo(input_tensor, input_size, anchor_boxes, yolohead_xyz_scales[0])

    data = torch.randn(3, 64, 256, 256)
    raw_boxes = torch.randn(3, 30, 7)
    label = torch.randn(3, 16, 16, 4, 6, 13)


    radarLoss = RadDetLoss(
        input_size=input_size,
        focal_loss_iou_threshold=focal_loss_iou_threshold
    )

    radarLoss(pred_raw=pred_raw,
              pred=pred,
              label=label,
              raw_boxes=raw_boxes[..., :6])
