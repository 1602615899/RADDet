# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import models.RADDet_finetune.helper as helper


def getTruePositive(pred, gt, input_size, iou_threshold=0.5, mode="3D"):
    """ output tp (true positive) with size [num_pred, ] """
    assert mode in ["3D", "2D"]
    tp = np.zeros(len(pred))
    detected_gt_boxes = []
    for i in range(len(pred)):
        current_pred = pred[i]
        if mode == "3D":
            current_pred_box = current_pred[:6]
            current_pred_score = current_pred[6]
            current_pred_class = current_pred[7]
            gt_box = gt[..., :6]
            gt_class = gt[..., 6]
        else:
            current_pred_box = current_pred[:4]
            current_pred_score = current_pred[4]
            current_pred_class = current_pred[5]
            gt_box = gt[..., :4]
            gt_class = gt[..., 4]

        if len(detected_gt_boxes) == len(gt): break

        if mode == "3D":
            iou = helper.iou3d(current_pred_box[np.newaxis, ...], gt_box, input_size)
        else:
            iou = helper.iou2d(current_pred_box[np.newaxis, ...], gt_box)
        iou_max_idx = np.argmax(iou)
        iou_max = iou[iou_max_idx]
        if iou_max >= iou_threshold and iou_max_idx not in detected_gt_boxes:
            tp[i] = 1.
            detected_gt_boxes.append(iou_max_idx)
    fp = 1. - tp
    return tp, fp
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def computeAP(tp, fp, num_gt_class):
    """ Compute Average Precision """
    tp_cumsum = np.cumsum(tp).astype(np.float32)
    fp_cumsum = np.cumsum(fp).astype(np.float32)
    recall = tp_cumsum / (num_gt_class + 1e-16)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    ########## NOTE: the following is under the reference of the repo ###########
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mAP(predictions, gts, input_size, ap_each_class, tp_iou_threshold=0.5, mode="3D"):
    """ Main function for calculating mAP
    Args:
        predictions         ->      [num_pred, 6 + score + class]
        gts                 ->      [num_gt, 6 + class]"""
    gts = gts[gts[..., :6].any(axis=-1) > 0]
    all_gt_classes = np.unique(gts[:, 6])
    ap_all = []
    # ap_all_classes = np.zeros(num_all_classes).astype(np.float32)
    for class_i in all_gt_classes:
        ### NOTE: get the prediction per class and sort it ###
        pred_class = predictions[predictions[..., 7] == class_i]
        pred_class = pred_class[np.argsort(pred_class[..., 6])[::-1]]
        ### NOTE: get the ground truth per class ###
        gt_class = gts[gts[..., 6] == class_i]
        tp, fp = getTruePositive(pred_class, gt_class, input_size, \
                                 iou_threshold=tp_iou_threshold, mode=mode)
        ap = computeAP(tp, fp, len(gt_class))
        ap_all.append(ap)
        ap_each_class[int(class_i)].append(ap)
    mean_ap = np.mean(ap_all)
    return mean_ap, ap_each_class


def mAP2D(predictions, gts, input_size, ap_each_class, tp_iou_threshold=0.5, mode="2D"):
    """ Main function for calculating mAP
    Args:
        predictions         ->      [num_pred, 4 + score + class]
        gts                 ->      [num_gt, 4 + class]"""
    gts = gts[gts[..., :4].any(axis=-1) > 0]
    all_gt_classes = np.unique(gts[:, 4])
    ap_all = []
    for class_i in all_gt_classes:
        ### NOTE: get the prediction per class and sort it ###
        pred_class = predictions[predictions[..., 5] == class_i]
        pred_class = pred_class[np.argsort(pred_class[..., 4])[::-1]]
        ### NOTE: get the ground truth per class ###
        gt_class = gts[gts[..., 4] == class_i]
        tp, fp = getTruePositive(pred_class, gt_class, input_size, \
                                 iou_threshold=tp_iou_threshold, mode=mode)
        ap = computeAP(tp, fp, len(gt_class))
        ap_all.append(ap)
        ap_each_class[int(class_i)].append(ap)
    mean_ap = np.mean(ap_all)
    return mean_ap, ap_each_class
