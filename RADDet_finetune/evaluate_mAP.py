# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
import cv2
from tabulate import tabulate
import pandas as pd

import torch.optim
from models.RADDet_finetune.RADDet_finetune_dataset import RADDet_Finetune
import models.RADDet_finetune.loader as loader
import argparse
import os
import sys
import numpy as np
# from torch.utils.data import DataLoader
import models.RADDet_finetune.mAP as mAP
from models.RADDet_finetune.yolo_head import decodeYolo, decodeYolo_fpn, yoloheadToPredictions, nms
import torch.distributed as dist


device = "cuda" if torch.cuda.is_available() else "cpu"

def cutImage(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1540:1750, :]
    part_2 = image[:, 2970:3550, :]
    part_3 = image[:, 4370:5400, :]
    part_4 = image[:, 6200:6850, :]
    new_img = np.concatenate([part_4, part_1, part_2, part_3], axis=1)
    cv2.imwrite(image_name, new_img)


def cutImage3Axes(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1780:2000, :]
    part_2 = image[:, 3800:4350, :]
    part_3 = image[:, 5950:6620, :]
    new_img = np.concatenate([part_3, part_1, part_2], axis=1)
    cv2.imwrite(image_name, new_img)


### NOTE: define testing step for RAD Boxes Model ###
def test_step(config_model, model, test_dataloader, num_classes, input_size,
              anchor_boxes, map_iou_threshold_list, device):
    mean_ap_test_all = []
    ap_all_class_test_all = []
    ap_all_class_all = []
    for i in range(len(map_iou_threshold_list)):
        mean_ap_test_all.append(0.0)
        ap_all_class_test_all.append([])
        ap_all_class = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        ap_all_class_all.append(ap_all_class)
    print("Start evaluating RAD Boxes on the test dataset, it might take a while...")
    # pbar = tqdm(total=int(data_generator.total_test_batches))
    for data, label0, label1, label2, raw_boxes in test_dataloader:
        data = data.to(device, non_blocking=True)
        data = data.permute(0, 3, 1, 2)
        data = data.unsqueeze(1)
        
        # print(data.shape)
        
        label0 = label0.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        raw_boxes = raw_boxes.to(device)

        feature0, feature1, feature2 = model(data)
        pred_raw0, pred0 = decodeYolo_fpn(feature0,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        pred_raw1, pred1 = decodeYolo_fpn(feature1,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
        pred_raw2, pred2 = decodeYolo_fpn(feature2,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
     
        pred0 = pred0.cpu().detach().numpy()
        pred1 = pred1.cpu().detach().numpy()
        pred2 = pred2.cpu().detach().numpy()
        raw_boxes = raw_boxes.cpu().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame0 = pred0[batch_id]
            pred_frame1 = pred1[batch_id]
            pred_frame2 = pred2[batch_id]
            predicitons0 = yoloheadToPredictions(pred_frame0,
                                                conf_threshold=config_model["confidence_threshold"])
            predicitons1 = yoloheadToPredictions(pred_frame1,
                                                conf_threshold=config_model["confidence_threshold"])
            predicitons2 = yoloheadToPredictions(pred_frame2,
                                                conf_threshold=config_model["confidence_threshold"])
            predicitons = np.concatenate((predicitons0, predicitons1, predicitons2), axis=0)
            nms_pred = nms(predicitons, config_model["nms_iou3d_threshold"],
                           config_model["input_shape"], sigma=0.3, method="nms")
            for j in range(len(map_iou_threshold_list)):
                map_iou_threshold = map_iou_threshold_list[j]
                mean_ap, ap_all_class_all[j] = mAP.mAP(nms_pred, raw_boxes_frame,
                                                       config_model["input_shape"],
                                                       ap_all_class_all[j],
                                                       tp_iou_threshold=map_iou_threshold)
                mean_ap_test_all[j] += mean_ap

    for iou_threshold_i in range(len(map_iou_threshold_list)):
        ap_all_class = ap_all_class_all[iou_threshold_i]
        mean_ap_test_all[iou_threshold_i], ap_all_class_test_all[iou_threshold_i] = dist_decouple_mAP(ap_all_class)
        # for ap_class_i in ap_all_class:
        #     if len(ap_class_i) == 0:
        #         class_ap = 0.
        #     else:
        #         class_ap = np.mean(ap_class_i)
        #     ap_all_class_test_all[iou_threshold_i].append(class_ap)
        # mean_ap_test_all[iou_threshold_i] = np.mean(ap_all_class_test_all[iou_threshold_i])
    return mean_ap_test_all, ap_all_class_test_all


@torch.no_grad()
def test_mAP(
    config_model,
    config_evaluate,
    config_data,
    model,
    device,
    test_loader,
    num_classes,
    anchor_boxes,
    output_dir,
    ):
    
    # switch to evaluation mode
    model.eval()
    
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)

    ### NOTE: evaluate RAD Boxes under different mAP_iou ###
    all_mean_aps, all_ap_classes = test_step(config_model=config_model,
                                             model=model,
                                             test_dataloader=test_loader,
                                             num_classes=num_classes,
                                             input_size=input_size,
                                             anchor_boxes=anchor_boxes,
                                             map_iou_threshold_list=config_evaluate["mAP_iou3d_threshold"],
                                             device=device
                                             )

    all_mean_aps = np.array(all_mean_aps)
    all_ap_classes = np.array(all_ap_classes)
    
    table = []
    row = []
    for i in range(len(all_mean_aps)):
        if i == 0:
            row.append("mAP")
        row.append(all_mean_aps[i])
    table.append(row)
    row = []
    for j in range(all_ap_classes.shape[1]):
        ap_current_class = all_ap_classes[:, j]
        for k in range(len(ap_current_class)):
            if k == 0:
                row.append(config_data["all_classes"][j])
            row.append(ap_current_class[k])
        table.append(row)
        row = []
    headers = []
    for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
        if ap_iou_i == 0:
            headers.append("AP name")
        headers.append("AP_%.2f"%(ap_iou_i))
    print("==================== RAD Boxes AP ========================")
    print(tabulate(table, headers=headers))
    print("==========================================================")
    filename = os.path.join(output_dir, config_data["RAD_dir"]+"-RAD Boxes AP.csv")
    df = pd.DataFrame(table, columns=['metric']+[ap for ap in headers])
    df.to_csv(filename, index=False, float_format="%.3f")


def dist_decouple_mAP(ap_all_class__):
    ap_all_class_test__ = []
    for ap_class_i in ap_all_class__:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = torch.tensor(np.sum(ap_class_i)).cuda()
            class_num = torch.tensor(len(ap_class_i)).cuda()
            dist.all_reduce(class_ap)
            dist.all_reduce(class_num)
            ap = class_ap / class_num
        ap_all_class_test__.append(ap.item())
    mean_ap_test__ = np.mean(ap_all_class_test__)
    
    return mean_ap_test__, ap_all_class_test__



def main(args, RAD_dir='RAD'):
    # initialization
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    # env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(env_str)

    config = loader.readConfig(config_file_name="./config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]

    config_data['RAD_dir'] = RAD_dir

    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./anchors.txt")
    num_classes = len(config_data["all_classes"])

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = RADDet(config_model, config_data, config_train, anchor_boxes)
    model.load_state_dict(torch.load(args.resume_from))
    model.to(device)

    test_dataset = RADDet_Finetune(config_data, config_train, config_model,
                                config_model["feature_out_shape"], anchor_boxes, dType="test", RADDir=RAD_dir)    # 2032
    
    if args.dist_eval:
        if len(test_dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) # Samples elements sequentially, always in the same order.    

    test_loader = DataLoader(test_dataset,
                             batch_size=config_train["batch_size"]//args.num_gpus,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,    # 这个参数为True可加快数据加载，但会占用更多内存

    )

    ### NOTE: RAD Boxes ckpt ###
    logdir = os.path.join(config_evaluate["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                          "-b_" + str(config_train["batch_size"]) +
                          "-lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    output_dir = os.path.join(config_evaluate["log_dir"], 'results_mAP')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)

    ### NOTE: evaluate RAD Boxes under different mAP_iou ###
    all_mean_aps, all_ap_classes = test_step(config_model=config_model,
                                             model=model,
                                             test_dataloader=test_loader,
                                             num_classes=num_classes,
                                             input_size=input_size,
                                             anchor_boxes=anchor_boxes,
                                             map_iou_threshold_list=config_evaluate["mAP_iou3d_threshold"]
                                             )

    all_mean_aps = np.array(all_mean_aps)
    all_ap_classes = np.array(all_ap_classes)

    table = []
    row = []
    for i in range(len(all_mean_aps)):
        if i == 0:
            row.append("mAP")
        row.append(all_mean_aps[i])
    table.append(row)
    row = []
    for j in range(all_ap_classes.shape[1]):
        ap_current_class = all_ap_classes[:, j]
        for k in range(len(ap_current_class)):
            if k == 0:
                row.append(config_data["all_classes"][j])
            row.append(ap_current_class[k])
        table.append(row)
        row = []
    headers = []
    for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
        if ap_iou_i == 0:
            headers.append("AP name")
        headers.append("AP_%.2f"%(ap_iou_i))
    print("==================== RAD Boxes AP ========================")
    print(tabulate(table, headers=headers))
    print("==========================================================")
    filename = os.path.join(output_dir, config_data["RAD_dir"]+"-RAD Boxes AP.csv")
    df = pd.DataFrame(table, columns=['metric']+[ap for ap in headers])
    df.to_csv(filename, index=False, float_format="%.3f")

    


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="The number of gpus.")
    parser.add_argument("--num-machines", type=int,
                        default=1,
                        help="The number of machines.")
    parser.add_argument("--machine-rank", type=int,
                        default=0,
                        help="The rank of current machine.")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist_url", type=str,
                        default="tcp://127.0.0.1:{}".format(port),
                        help="initialization URL for pytorch distributed backend.")
    parser.add_argument("--resume_from", type=str,
                        default="/mnt/SrvUserDisk/ZhangXu/RADAR/RADDet_Pytorch/logs/RadarResNet/Normalize_to_[0,1]-RAD-b_6-lr_0.0005/ckpt/best.pth",
                        help="The number of machines.")
    parser.add_argument("--cart_resume_from", type=str,
                        default="/mnt/SrvUserDisk/ZhangXu/RADAR/RADDet_Pytorch/logs/RadarResNet/Normalize_to_[0,1]-RAD-b_6-lr_0.0005_cartesian/ckpt/best.pth",
                        help="The number of machines.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    # RAD_dir = 'RAD', 'RAD4', 'RAD_Sim8'
    main(args, RAD_dir='RAD')
    # main(args, RAD_dir='RAD4')
    # main(args, RAD_dir='RAD_Sim8')

