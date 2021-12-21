#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:19:20 2021

@author: thomas_yang
"""

# import tensorflow as tf
import cv2
import numpy as np
import config
# import sys

def idx2class():
    return dict((v, k) for k, v in config.classes_label.items())
  

def draw_boxes_joint_on_image(image, boxes, heatmap_hand_tmp, scores, classes):

    color=[(0, 255, 0), (0, 0, 255)]
    # load label index/name
    idx2class_dict = idx2class()
    boxes = boxes.astype(np.int)
    for i in range(boxes.shape[0]):   
        if heatmap_hand_tmp[i, 0] >  heatmap_hand_tmp[i, 1]:
            clases = 0
        else:
            clases = 1
            
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[clases]), scores[i])
        
        # draw bbox        
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=color[int(clases)], thickness=2)

        # label class name
        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
                    
    return image