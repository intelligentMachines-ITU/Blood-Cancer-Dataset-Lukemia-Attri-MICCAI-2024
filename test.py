# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import csv
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    get_fixed_xyxy
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
from utils.my_model import MyCNN
import torch.nn.functional as F
import re
from torchvision.ops import roi_align
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate




def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )

def my_process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix and top indices.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
        iouv (tensor[10]), IoU thresholds
    Returns:
        correct (array[N, 10]), for 10 IoU levels
        top_indices (tensor[M]), top IoU-gaining detection indices for each label
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    
    top_indices = torch.argmax(iou, dim=1)

    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), top_indices



def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    all_rows = []

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            (preds,int_feats),_  = model(im) if compute_loss else (model(im, augment=augment), None) 
            train_out = preds[1]
            in_channels = int_feats[0].shape[1]
            attribute_model_path =  ROOT / "Attribute_model/last_weights.pth"
            custom_weights = torch.load(attribute_model_path)
            cell_model = MyCNN(num_classes=12, dropout_prob=0.5, in_channels=480).to(device)
            cell_model.load_state_dict(custom_weights)



        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        attribute_targets = targets[:,7:13]
        targets = targets[:, 0:6] # I changed here    
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        
        # Metrics
        for si, pred in enumerate(preds):

            labels = targets[targets[:, 0] == si, 1:]
            all_top_indices_cell_pred = []
            pred_Nuclear_Chromatin_array = []
            pred_Nuclear_Shape_array = []
            pred_Nucleus_array = []
            pred_Cytoplasm_array = []
            pred_Cytoplasmic_Basophilia_array = []
            pred_Cytoplasmic_Vacuoles_array = []

            if (len(pred)>0):
                boxes = torch.cat([si * torch.ones(pred.shape[0], 1).to(device), pred[:, 0:4].to(device)], dim=1)


                int_feats_p2 = int_feats[0][si].to(torch.float32).unsqueeze(0)
                int_feats_p3 = int_feats[1][si].to(torch.float32).unsqueeze(0)


                torch.cuda.empty_cache()


                # del int_feats
                all_top_indices_cell_pred = []

                for i in range(len(pred)):

                    pred_tensor = pred[i, 0:4]
                    img_shape_tensor = torch.tensor([im.shape[2], im.shape[3],im.shape[2],im.shape[3]]).to(device)

                    normalized_xyxy=pred_tensor / img_shape_tensor
                    p2_feature_shape_tensor = torch.tensor([int_feats[0][si].shape[1], int_feats[0][si].shape[2],int_feats[0][si].shape[1],int_feats[0][si].shape[2]]).to(device)                        # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)
                    p3_feature_shape_tensor = torch.tensor([int_feats[1][si].shape[1], int_feats[1][si].shape[2],int_feats[1][si].shape[1],int_feats[1][si].shape[2]]).to(device)             # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)

                    
                    p2_normalized_xyxy = normalized_xyxy*p2_feature_shape_tensor
                    p3_normalized_xyxy = normalized_xyxy*p3_feature_shape_tensor

                    
                    p2_x_min, p2_y_min, p2_x_max, p2_y_max = get_fixed_xyxy(p2_normalized_xyxy,int_feats_p2)
                    p3_x_min, p3_y_min, p3_x_max, p3_y_max = get_fixed_xyxy(p3_normalized_xyxy,int_feats_p3)


                    p2_roi = torch.tensor([p2_x_min, p2_y_min, p2_x_max, p2_y_max], device=device).float() 
                    p3_roi = torch.tensor([p3_x_min, p3_y_min, p3_x_max, p3_y_max], device=device).float() 

                    batch_index = torch.tensor([0], dtype=torch.float32, device = device)

                    # Concatenate the batch index to the bounding box coordinates
                    p2_roi_with_batch_index = torch.cat([batch_index, p2_roi])
                    p3_roi_with_batch_index = torch.cat([batch_index, p3_roi])


                    p2_resized_object = roi_align(int_feats_p2, p2_roi_with_batch_index.unsqueeze(0).to(device), output_size=(24, 30))
                    p3_resized_object = roi_align(int_feats_p3, p3_roi_with_batch_index.unsqueeze(0).to(device), output_size=(24, 30))
                    concat_box = torch.cat([p2_resized_object,p3_resized_object],dim=1)

                    in_channels = concat_box.size(1)
                    cell_model.eval().to(device)
                    output_cell_prediction= cell_model(concat_box)
                    output_cell_prediction_prob = F.softmax(output_cell_prediction.view(6,2), dim=1)
                    top_indices_cell_pred = torch.argmax(output_cell_prediction_prob, dim=1)
                    all_top_indices_cell_pred.append(top_indices_cell_pred)
                    pred_Nuclear_Chromatin_array.append(top_indices_cell_pred[0].item())
                    pred_Nuclear_Shape_array.append(top_indices_cell_pred[1].item())
                    pred_Nucleus_array.append(top_indices_cell_pred[2].item())
                    pred_Cytoplasm_array.append(top_indices_cell_pred[3].item())
                    pred_Cytoplasmic_Basophilia_array.append(top_indices_cell_pred[4].item())
                    pred_Cytoplasmic_Vacuoles_array.append(top_indices_cell_pred[5].item())



            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        # conf_,indices=my_process_batch(detections=None, labels=labels[:, 0])
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                _,max_iou_indices = my_process_batch(predn, labelsn, iouv)
                max_iou_predicted_boxes=boxes[max_iou_indices]
                attributes_prediction=[]
                epoch = 0
                for i in range(len(max_iou_indices)):

                    attributes_pred = all_top_indices_cell_pred[max_iou_indices[i]].detach().cpu()
                    attributes_prediction.append(attributes_pred)
                
                path_object = Path(path)
                # Extracting the name
                file_name = path_object.name
                filenames =[]

                # Initialize lists for each attribute
                attribute_names = ['Nuclear Chromatin', 'Nuclear Shape', 'Nucleus', 'Cytoplasm', 'Cytoplasmic Basophilia', 'Cytoplasmic Vacuoles']

                all_labels_list = [[] for _ in range(6)]
                all_prediction_attributes_list = [[] for _ in range(6)]
  


                for i in range(len(attributes_prediction)):
                    count = 0
                    label = attribute_targets[i].detach().cpu().int()

                    if all(x != 2 for x in label):
                        filenames.append(file_name)

                        # Append the label and prediction to the corresponding attribute list
                        for j in range(6):
                            all_labels_list[j].append(label[j].item())
                            all_prediction_attributes_list[j].append(attributes_prediction[i][j].item())

                # Combine data into rows
                rows = zip(filenames, *all_labels_list, *all_prediction_attributes_list)
                all_rows.extend(rows)

                if plots:
                    cnf_,indices= my_process_batch(predn, labelsn,iouv=iouv)
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)


    attribute_model_dir = ROOT / "Attribute_model/test/test"
    attribute_names = ['Nuclear Chromatin', 'Nuclear Shape', 'Nucleus', 'Cytoplasm', 'Cytoplasmic Basophilia', 'Cytoplasmic Vacuoles']

    # Create directory if it doesn't exist
    attribute_model_dir.mkdir(parents=True, exist_ok=True)

                # Write to the CSV file
    csv_file_path = f"{attribute_model_dir}.csv"
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

                # Write header if the file is empty
        if os.path.getsize(csv_file_path) == 0:
            header = ['filename'] + [f'{attr}' for attr in attribute_names] + [f'pred_{attr}' for attr in attribute_names]
            csv_writer.writerow(header)

                    # Write all accumulated rows
        csv_writer.writerows(all_rows)


    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, header=None, names=['filename', 'Nuclear Chromatin','Nuclear Shape','Nucleus','Cytoplasm','Cytoplasmic Basophilia','Cytoplasmic Vacuoles','pred_Nuclear Chromatin','pred_Nuclear Shape','pred_Nucleus','pred_Cytoplasm','pred_Cytoplasmic Basophilia','pred_Cytoplasmic Vacuoles'])

    # Drop the first row if it contains headers
    df = df.iloc[1:]

    # Convert label columns to numeric, replace invalid literals with NaN
    Nuclear_Chromatin_array = pd.to_numeric(df['Nuclear Chromatin'], errors='coerce')
    Nuclear_Shape_array = pd.to_numeric(df['Nuclear Shape'], errors='coerce')
    Nucleus_array = pd.to_numeric(df['Nucleus'], errors='coerce')
    Cytoplasm_array = pd.to_numeric(df['Cytoplasm'], errors='coerce')
    Cytoplasmic_Basophilia_array = pd.to_numeric(df['Cytoplasmic Basophilia'], errors='coerce')
    Cytoplasmic_Vacuoles_array = pd.to_numeric(df['Cytoplasmic Vacuoles'], errors='coerce')

    pred_Nuclear_Chromatin_array = pd.to_numeric(df['pred_Nuclear Chromatin'], errors='coerce')
    pred_Nuclear_Shape_array = pd.to_numeric(df['pred_Nuclear Shape'], errors='coerce')
    pred_Nucleus_array = pd.to_numeric(df['pred_Nucleus'], errors='coerce')
    pred_Cytoplasm_array = pd.to_numeric(df['pred_Cytoplasm'], errors='coerce')
    pred_Cytoplasmic_Basophilia_array = pd.to_numeric(df['pred_Cytoplasmic Basophilia'], errors='coerce')
    pred_Cytoplasmic_Vacuoles_array = pd.to_numeric(df['pred_Cytoplasmic Vacuoles'], errors='coerce')


    # Exclude or replace non-numeric entries
    Nuclear_Chromatin_array = Nuclear_Chromatin_array[~np.isnan(Nuclear_Chromatin_array)].astype(int)
    Nuclear_Shape_array = Nuclear_Shape_array[~np.isnan(Nuclear_Shape_array)].astype(int)
    Nucleus_array = Nucleus_array[~np.isnan(Nucleus_array)].astype(int)
    Cytoplasm_array = Cytoplasm_array[~np.isnan(Cytoplasm_array)].astype(int)
    Cytoplasmic_Basophilia_array = Cytoplasmic_Basophilia_array[~np.isnan(Cytoplasmic_Basophilia_array)].astype(int)
    Cytoplasmic_Vacuoles_array = Cytoplasmic_Vacuoles_array[~np.isnan(Cytoplasmic_Vacuoles_array)].astype(int)

    # Exclude or replace non-numeric entries
    pred_Nuclear_Chromatin_array = pred_Nuclear_Chromatin_array[~np.isnan(pred_Nuclear_Chromatin_array)].astype(int)
    pred_Nuclear_Shape_array = pred_Nuclear_Shape_array[~np.isnan(pred_Nuclear_Shape_array)].astype(int)
    pred_Nucleus_array = pred_Nucleus_array[~np.isnan(pred_Nucleus_array)].astype(int)
    pred_Cytoplasm_array = pred_Cytoplasm_array[~np.isnan(pred_Cytoplasm_array)].astype(int)
    pred_Cytoplasmic_Basophilia_array = pred_Cytoplasmic_Basophilia_array[~np.isnan(pred_Cytoplasmic_Basophilia_array)].astype(int)
    pred_Cytoplasmic_Vacuoles_array = pred_Cytoplasmic_Vacuoles_array[~np.isnan(pred_Cytoplasmic_Vacuoles_array)].astype(int)

    # print(f"\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3}, F1: {f1:.3}")
    def compute_and_print_metrics(attribute_name, true_labels, pred_labels):
        # Exclude or replace non-numeric entries
        true_labels = true_labels[~np.isnan(true_labels)].astype(int)
        pred_labels = pred_labels[~np.isnan(pred_labels)].astype(int)

        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)

        return [attribute_name, accuracy, precision, recall, f1]



    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    metrics = []
    NC_prec = compute_and_print_metrics('Nuclear_Chromatin', Nuclear_Chromatin_array, pred_Nuclear_Chromatin_array)
    NS_prec = compute_and_print_metrics('Nuclear_Shape', Nuclear_Shape_array, pred_Nuclear_Shape_array)
    N_prec = compute_and_print_metrics('Nucleus', Nucleus_array, pred_Nucleus_array)
    C_prec = compute_and_print_metrics('Cytoplasm', Cytoplasm_array, pred_Cytoplasm_array)
    CB_prec = compute_and_print_metrics('Cytoplasmic_Basophilia', Cytoplasmic_Basophilia_array, pred_Cytoplasmic_Basophilia_array)
    CV_prec = compute_and_print_metrics('Cytoplasmic_Vacuoles', Cytoplasmic_Vacuoles_array, pred_Cytoplasmic_Vacuoles_array)

    # Calculate average precision
    average_precision = np.mean([NC_prec[-1], NS_prec[-1], N_prec[-1], C_prec[-1], CB_prec[-1], CV_prec[-1]])

    # Append results to metrics list
    metrics.extend([NC_prec, NS_prec, N_prec, C_prec, CB_prec, CV_prec])

    # Print table
    headers = ["Attribute", "Accuracy", "Precision", "Recall", "F1"]
    print(tabulate(metrics, headers=headers, tablefmt="grid"))  


    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/WBC_v1.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5x_300Epochs_training/weights/best.pt', help='model path or triton URL')
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="test", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)