import cv2
import numpy as np
import os
from yolov7 import YOLOv7
from PIL import Image, ImageOps
import pandas as pd
import csv


def detect_object(frame):
    model_path = "models/best.onnx"
    yolov7_detector = YOLOv7(model_path, conf_thres=0.2, iou_thres=0.3)
    boxes, scores, class_ids = yolov7_detector(frame)
    combined_img = yolov7_detector.draw_detections(frame)
    
    return combined_img

def detect_dic(frame):
    model_path = "models/best.onnx"
    yolov7_detector = YOLOv7(model_path, conf_thres=0.2, iou_thres=0.3)
    boxes, scores, class_ids = yolov7_detector(frame)

    with open('coco.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    
    xmin = []
    ymin = []    
    xmax = []
    ymax = []
    class_id = []
    
    for classId, confidence, box in zip(class_ids.flatten(), scores.flatten(), boxes):
        xmin_, ymin_, xmax_, ymax_ = box
        xmin.append(xmin_)
        ymin.append(ymin_)
        xmax.append(xmax_)
        ymax.append(ymax_)
        class_id.append(names[classId])
    
    detection_dic = {'class_id' : class_id, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax}
        
    return detection_dic