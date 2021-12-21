#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:38:49 2021

@author: thomas_yang
"""



import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as trns
from PIL import Image


_GRAY = (218, 227, 218)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_WHITE = (255, 255, 255)

keypoints = ['nose','left_eye','right_eye',
             'left_ear','right_ear','left_shoulder',
             'right_shoulder','left_elbow','right_elbow',
             'left_wrist','right_wrist','left_hip',
             'right_hip','left_knee', 'right_knee', 
             'left_ankle','right_ankle']


_COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def vis_bbox(image, bbox, color=_GREEN, thick=1):
    """Visualizes a bounding box."""
    image = image.astype(np.uint8)
    bbox = list(map(int, bbox))
    x0, y0, x1, y1 = bbox
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=thick)
    return image


def vis_mask(image, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""
    image = image.astype(np.float32)

    mask = mask >= 0.5
    mask = mask.astype(np.uint8)
    idx = np.nonzero(mask)

    image[idx[0], idx[1], :] *= 1.0 - alpha
    image[idx[0], idx[1], :] += alpha * col

    if show_border:
        contours = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(image, contours, -1, _WHITE,
                         border_thick, cv2.LINE_AA)

    return image.astype(np.uint8)


def vis_class(image, bbox, text, bg_color=_GREEN, text_color=_GRAY, font_scale=0.35):
    """Visualizes the class."""
    image = image.astype(np.uint8)
    x0, y0 = int(bbox[0]), int(bbox[1])

    # Compute text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)

    # Place text background
    back_tl = x0, y0 - int(1.3 * text_h)
    back_br = x0 + text_w, y0
    cv2.rectangle(image, back_tl, back_br, bg_color, -1)

    # Show text
    text_tl = x0, y0 - int(0.3 * text_h)
    cv2.putText(image, text, text_tl, font, font_scale,
                text_color, lineType=cv2.LINE_AA)

    return image


def run_object_detection(model, image_path, transforms, threshold=0.4):
    """Inference."""

    # Read image and run prepro
    image = Image.open(image_path).convert("RGB")
    # image = image.resize((480, 270))
   
    image_tensor = transforms(image).cuda()
    outputs = model([image_tensor])[0]
    # print(outputs)

    # # Result postpro and vis   
    # display_image = np.array(image)    
    # outputs = {k: v.cpu().numpy() for k, v in outputs.items()}

    # for i, (bbox, label, score) in enumerate(zip(outputs["boxes"], outputs["labels"], outputs["scores"])):
    #     if score < threshold or label != 1:
    #         continue
        
    #     print(f"Label {label}: {_COCO_INSTANCE_CATEGORY_NAMES[label]} ({score:.2f})")
    #     print(bbox)
    #     display_image = vis_bbox(display_image, bbox)
    #     display_image = vis_class(display_image, bbox, _COCO_INSTANCE_CATEGORY_NAMES[label])
    
    # cv2.imshow('display_image', display_image[...,::-1])
    # cv2.waitKey(1)

    return outputs

def draw_keypoints_per_person(img, image_path, all_keypoints, all_scores, confs, boxes, keypoint_threshold=2, conf_threshold=0.965):
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # pick a set of N color-ids from the spectrum
    color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
    
    # iterate for every person detected
    txt_path = image_path.replace('images', 'labels_with_ids').replace('.jpg', '.txt')
    with open(txt_path, 'w') as f:
        for person_id in range(len(all_keypoints)):
          # check the confidence score of the detected person
          if confs[person_id]>conf_threshold:
            
            bbox = list(map(int, boxes[person_id]))
            if (bbox[2] - bbox[0])* (bbox[3] - bbox[1]) < (25*25):
                continue
                        
            keypoints = all_keypoints[person_id, ...]
            scores = all_scores[person_id, ...]
            is_rasie_hand = '0'
            for kp in range(len(scores)):
                if scores[kp]>keypoint_threshold:
                    keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                    color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                    if kp == 9 or kp == 10:
                        cv2.circle(img_copy, keypoint, 2, (0, 255, 255), -1)                    
                    else:
                        cv2.circle(img_copy, keypoint, 2, color, -1)
    
            img_copy = vis_bbox(img_copy, boxes[person_id], color=_GREEN, thick=1)
            keypoint_0 = tuple(map(int, keypoints[0, :2].detach().numpy().tolist()))
            keypoint_3 = tuple(map(int, keypoints[3, :2].detach().numpy().tolist()))
            keypoint_4 = tuple(map(int, keypoints[4, :2].detach().numpy().tolist()))
            keypoint_7 = tuple(map(int, keypoints[7, :2].detach().numpy().tolist()))
            keypoint_8 = tuple(map(int, keypoints[8, :2].detach().numpy().tolist()))
            keypoint_9 = tuple(map(int, keypoints[9, :2].detach().numpy().tolist()))
            keypoint_10 = tuple(map(int, keypoints[10, :2].detach().numpy().tolist()))

            if scores[0]>keypoint_threshold and scores[3]>keypoint_threshold and scores[4]>keypoint_threshold and scores[7]>keypoint_threshold:                
                if keypoint_7[1] < keypoint_0[1] and keypoint_7[1] < keypoint_3[1] and keypoint_7[1] < keypoint_4[1]:
                    img_copy = vis_bbox(img_copy, boxes[person_id], color=_RED, thick=1) 
                    is_rasie_hand = '1'            
            if scores[0]>keypoint_threshold and scores[3]>keypoint_threshold and scores[4]>keypoint_threshold and scores[8]>keypoint_threshold:
                if keypoint_8[1] < keypoint_0[1] and keypoint_8[1] < keypoint_3[1] and keypoint_8[1] < keypoint_4[1]:
                    img_copy = vis_bbox(img_copy, boxes[person_id], color=_RED, thick=1) 
                    is_rasie_hand = '1'
            
            if scores[0]>keypoint_threshold and scores[3]>keypoint_threshold and scores[4]>keypoint_threshold and scores[9]>keypoint_threshold:                
                if keypoint_9[1] < keypoint_0[1] and keypoint_9[1] < keypoint_3[1] and keypoint_9[1] < keypoint_4[1]:
                    img_copy = vis_bbox(img_copy, boxes[person_id], color=_RED, thick=1) 
                    is_rasie_hand = '1'            
            if scores[0]>keypoint_threshold and scores[3]>keypoint_threshold and scores[4]>keypoint_threshold and scores[10]>keypoint_threshold:
                if keypoint_10[1] < keypoint_0[1] and keypoint_10[1] < keypoint_3[1] and keypoint_10[1] < keypoint_4[1]:
                    img_copy = vis_bbox(img_copy, boxes[person_id], color=_RED, thick=1) 
                    is_rasie_hand = '1'
            
            bbox = boxes[person_id].cpu().detach().numpy() / (img_copy.shape[1], img_copy.shape[0], img_copy.shape[1], img_copy.shape[0])
            bbox = list(map(str, bbox))
            # bbox = [i.detach().numpy() for i in bbox]
            info = is_rasie_hand +' ' + '-1'  + ' ' + ' '.join(bbox) + '\n'
            f.write(info)

    return img_copy


if __name__ == "__main__":

    image_path = '/home/thomas_yang/ML/datasets/RaiseHand/confer_2021_1211_15/images/'
    image_list = [image_path + i for i in os.listdir(image_path)]
    image_list.sort()
    transforms = trns.ToTensor()

    # Load model
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

    # Set model to eval mode
    model.eval().cuda()

    for img_idx, image_path in enumerate(image_list[0:]):
        print(img_idx, image_path.split('/')[-1])
        image = Image.open(image_path).convert("RGB")
        image = image.resize((960, 540))
        image_np = np.array(image)
       
        image_tensor = transforms(image).cuda()
        with torch.no_grad():
            output = model([image_tensor])[0]
        
        for key in output:
            output[key] = output[key].cpu()
        
        print(output["scores"])
        keypoints_img = draw_keypoints_per_person(image_np,
                                                  image_path,
                                                  output["keypoints"], 
                                                  output["keypoints_scores"], 
                                                  output["scores"], 
                                                  output['boxes'], 
                                                  keypoint_threshold=2)    

        cv2.imshow('keypoints_img', keypoints_img[...,::-1])
        cv2.waitKey(1)            
            
            
        
        
  
        
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    