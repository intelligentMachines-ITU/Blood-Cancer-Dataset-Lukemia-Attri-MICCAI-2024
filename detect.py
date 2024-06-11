# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
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
import csv
import os
import platform
import sys
from pathlib import Path

import torch
import copy
import torch.nn.functional as F


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,get_fixed_xyxy)
from utils.torch_utils import select_device, smart_inference_mode
from utils.my_model import MyCNN
from torchvision.ops import roi_align

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # stride = 16
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s, orig_img in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred,int_feats = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred, int_feats = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred,int_feats = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
            # int_feats_p3= int_feats[0][0,:,:,:].to(torch.float32)
            # int_feats_p3 = int_feats_p3.unsqueeze(0)#.unsqueeze(0)

            int_feats_p2 = int_feats[0][0].to(torch.float32).unsqueeze(0)
            int_feats_p3 = int_feats[1][0].to(torch.float32).unsqueeze(0)

            # concat_feat = torch.cat([int_feats_p2,int_feats_p3],dim=1)
            in_channels = int_feats_p2.shape[1]+int_feats_p3.shape[1]
            cell_attribute_model= MyCNN(num_classes=12, dropout_prob=0.5, in_channels=in_channels).to(device)
            folder_name = 'data/WBC_dataset_sample/Attribute_model'
            custom_weights_path = f"{folder_name}/last_weights.pth"
            custom_weights = torch.load(custom_weights_path)
            cell_attribute_model.load_state_dict(custom_weights)
            cell_attribute_model.eval().to(device)

            # int_feats_p5= int_feats[1][0,:,:,:].to(torch.float32)
            # int_feats_p5 = int_feats_p5.unsqueeze(0)#.unsqueeze(0)
            torch.cuda.empty_cache()

                # del int_feats
            # resized_int_feats_p5 = F.interpolate(int_feats_p5, size=(int_feats[0].size(2), int_feats[0].size(3)), mode='bilinear', align_corners=False)
            # concatenated_features = torch.cat([resized_int_feats_p5,int_feats_p3],dim=1)
            
            if (len(pred)>0):
                all_top_indices_cell_pred = []
                top_indices_cell_pred = []
                pred_Nuclear_Chromatin_array = []
                pred_Nuclear_Shape_array = []
                pred_Nucleus_array = []
                pred_Cytoplasm_array = []
                pred_Cytoplasmic_Basophilia_array = []
                pred_Cytoplasmic_Vacuoles_array = []

                for i in range(len(pred[0])):
                    if pred[0][i].numel() > 0:  # Check if the tensor is not empty

                        pred_tensor = pred[0][i][0:4]
                        
                        if pred[0][i][5] != 0:
                            
                            img_shape_tensor = torch.tensor([im.shape[2], im.shape[3],im.shape[2],im.shape[3]]).to(device)

                            normalized_xyxy=pred_tensor / img_shape_tensor
                            p2_feature_shape_tensor = torch.tensor([int_feats[0].shape[1], int_feats[0].shape[2],int_feats[0].shape[1],int_feats[0].shape[2]]).to(device)                        # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)
                            p3_feature_shape_tensor = torch.tensor([int_feats[1].shape[1], int_feats[1].shape[2],int_feats[1].shape[1],int_feats[1].shape[2]]).to(device)             # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)
                        
                        
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

                            output_cell_prediction= cell_attribute_model(concat_box)
                            output_cell_prediction_prob = F.softmax(output_cell_prediction.view(6,2), dim=1)
                            top_indices_cell_pred = torch.argmax(output_cell_prediction_prob, dim=1)
                            pred_Nuclear_Chromatin_array.append(top_indices_cell_pred[0].item())
                            pred_Nuclear_Shape_array.append(top_indices_cell_pred[1].item())
                            pred_Nucleus_array.append(top_indices_cell_pred[2].item())
                            pred_Cytoplasm_array.append(top_indices_cell_pred[3].item())
                            pred_Cytoplasmic_Basophilia_array.append(top_indices_cell_pred[4].item())
                            pred_Cytoplasmic_Vacuoles_array.append(top_indices_cell_pred[5].item())
                        # all_top_indices_cell_pred.append(top_indices_cell_pred.item())
                        else:
                            # top_indices_cell_pred = torch.tensor([0,0,0,0,0,0]).to(device)
                            pred_Nuclear_Chromatin_array.append(0)
                            pred_Nuclear_Shape_array.append(0)
                            pred_Nucleus_array.append(0)
                            pred_Cytoplasm_array.append(0)
                            pred_Cytoplasmic_Basophilia_array.append(0)
                            pred_Cytoplasmic_Vacuoles_array.append(0)




        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # # Create or append to the CSV file
        # def write_to_csv(name, predicts, confid,pred_NC,pred_NS, 
        #                  pred_N,pred_C,pred_CB,
        #                  pred_CV,x_min,y_min,x_max,y_max):
        #     data = {'Image Name': name, 'Prediction': predicts, 'Confidence': confid, 'Nuclear Chromatin':pred_NC,
        #             'Nuclear Shape':pred_NS,'Nucleus':pred_N,'Cytoplasm':pred_C,
        #             'Cytoplasmic Basophilia': pred_CB, 'Cytoplasmic Vacuoles': pred_CV,
        #             'x_min':x_min,'y_min':y_min,'x_max':x_max,'y_max':y_max}
        #     with open(csv_path, mode='a', newline='') as f:
        #         writer = csv.DictWriter(f, fieldnames=data.keys())
        #         if not csv_path.is_file():
        #             writer.writeheader()
        #         writer.writerow(data)
        # Create or append to the CSV file
        def write_to_csv(name, predicts, confid, pred_NC, pred_NS, 
                        pred_N, pred_C, pred_CB, pred_CV,
                        x_min, y_min, x_max, y_max):
            data = {'Image Name': name, 'Prediction': predicts, 'Confidence': confid, 'Nuclear Chromatin': pred_NC,
                    'Nuclear Shape': pred_NS, 'Nucleus': pred_N, 'Cytoplasm': pred_C,
                    'Cytoplasmic Basophilia': pred_CB, 'Cytoplasmic Vacuoles': pred_CV,
                    'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
            
            # Check if the CSV file exists
            if not os.path.isfile(csv_path):
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    writer.writeheader()

            # Append data to CSV file
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for count, (*xyxy, conf, cls) in enumerate(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'
                    
                    if save_csv:
                        x_min,y_min,x_max,y_max = xyxy

                        # Scaling factors
                        scale_width = orig_img.shape[1] / 640
                        scale_height = orig_img.shape[0] / 640

                        # Convert bounding box coordinates to 800x448 image
                        x_min_new = int(x_min * scale_width)
                        y_min_new = int(y_min * scale_height)
                        x_max_new = int(x_max * scale_width)
                        y_max_new = int(y_max * scale_height)

                        write_to_csv(p.name, label, confidence_str,
                                     pred_Nuclear_Chromatin_array[count],pred_Nuclear_Shape_array[count], 
                                     pred_Nucleus_array[count],pred_Cytoplasm_array[count],pred_Cytoplasmic_Basophilia_array[count],
                                     pred_Cytoplasmic_Vacuoles_array[count],
                                     int(x_min_new),int(y_min_new),
                                     int(x_max_new),int(y_max_new))
                        

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.my_box_label(xyxy, label, color=colors(c, True), att1=pred_Nuclear_Chromatin_array[0],
                        #                        att2 = pred_Nuclear_Shape_array[0], att3 = pred_Nucleus_array[0],
                        #                        att4 = pred_Cytoplasm_array[0], att5 = pred_Cytoplasmic_Basophilia_array[0],
                        #                        att6 = pred_Cytoplasmic_Vacuoles_array[0]
                        #                        )

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5x_300Epochs_training/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='/home/iml/Desktop/bc_experiment/HCM_V3/HCM_840_attribute/images/test/', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/WBC_v1.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
