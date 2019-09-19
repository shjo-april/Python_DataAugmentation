# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import glob
import random
import numpy as np

import xml.etree.ElementTree as ET

from DataAugmentation import *

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", 
               "diningtable", "dog", "horse", "motorbike", "person", 
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}

def xml_read(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-3] + '*'
    image_path = image_path.replace('/xml', '/image')
    image_path = glob.glob(image_path)[0]

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(CLASS_DIC[label])

    return image_path, np.asarray(bboxes, dtype = np.float32), np.asarray(classes, dtype = np.int32)

# main
xml_paths = glob.glob('./VOC2007/xml/*.xml')

augment_names = ['horizental_flip', 'vertical_flip', 'scale', 'hue', 'saturation', 'gray', 'brightness', 'gaussian', 'shift', 'crop']

for xml_path in xml_paths:
    image_path, _gt_bboxes, _gt_classes = xml_read(xml_path)

    _image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)

    h_image, h_gt_bboxes = random_horizental_flip(_image.copy(), _gt_bboxes.copy(), threshold = 1.0)
    v_image, v_gt_bboxes = random_vertical_flip(_image.copy(), _gt_bboxes.copy(), threshold = 1.0)
    s_image, s_gt_bboxes = random_scale(_image.copy(), _gt_bboxes.copy(), threshold = 1.0)
    hue_image = random_hue(_image.copy(), threshold = 1.0)
    sat_image = random_saturation(_image.copy(), threshold = 1.0)
    gray_image = random_gray(_image.copy(), threshold = 1.0)
    b_image = random_brightness(_image.copy(), threshold = 1.0)
    g_image = random_gaussian_noise(_image.copy(), threshold = 1.0)
    shift_image, shift_gt_bboxes = random_shift(_image.copy(), _gt_bboxes.copy(), threshold = 1.0)
    crop_image, crop_gt_bboxes, crop_gt_classes = random_crop(_image.copy(), _gt_bboxes.copy(), _gt_classes.copy(), threshold = 1.0)

    augment_dic = {
        'horizental_flip' : [h_image, h_gt_bboxes, _gt_classes],
        'vertical_flip' : [v_image, v_gt_bboxes, _gt_classes],
        'scale' : [s_image, s_gt_bboxes, _gt_classes],
        'hue' : [hue_image, _gt_bboxes, _gt_classes],
        'saturation' : [sat_image, _gt_bboxes, _gt_classes],
        'gray' : [gray_image, _gt_bboxes, _gt_classes],
        'brightness' : [b_image, _gt_bboxes, _gt_classes],
        'gaussian' : [g_image, _gt_bboxes, _gt_classes],
        'shift' : [shift_image, shift_gt_bboxes, _gt_classes],
        'crop' : [crop_image, crop_gt_bboxes, crop_gt_classes],
    }

    image_w_size = 640
    image_h_size = 480
    
    for i, augment_name in enumerate(augment_names):
        image, gt_bboxes, gt_classes = augment_dic[augment_name]

        for gt_bbox, gt_class in zip(gt_bboxes, gt_classes):
            xmin, ymin, xmax, ymax = gt_bbox[:4].astype(np.int32)

            # Text
            string = "{}".format(CLASS_NAMES[gt_class])
            text_size = cv2.getTextSize(string, 0, 0.5, thickness = 1)[0]

            cv2.rectangle(image, (xmin, ymin), (xmin + text_size[0], ymin - text_size[1] - 5), (0, 255, 0), -1)
            cv2.putText(image, string, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType = cv2.LINE_AA)
            
            # Rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        image = cv2.resize(image.astype(np.uint8), (image_w_size, image_h_size), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('./res/{}_{}.jpg'.format(image_name, augment_name), image)
    