#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:08:21 2021

@author: thomas_yang
"""

import torch
import cv2
import numpy as np
from decoder import Decoder
from resnetFPN import MyResNet50
from draw import draw_boxes_joint_on_image
import config

device = torch.device('cuda')
model = MyResNet50().cuda()
model.load_state_dict(torch.load('./saved_model/resnet50FPN_256_epoch_{}.pth'.format(config.finetune_load_epoch)))
model.eval()
on_video = True

#%%
if on_video:
    
    # from configuration import Config
    # import collections
    import os
    # video_path = '/home/thomas_yang/Downloads/2021-09-03-kaohsiung-5g-base-vlc-record/'
    # video_path = '/home/thomas_yang/ML/datasets/RaiseHand/vlc-record-2021-11-29-15h41m44s-rtsp___192.168.0.100_8554_fhd-/'
    # video_path = '/home/thomas_yang/ML/datasets/RaiseHand/testing_data/'
    video_path = '/home/thomas_yang/ML/datasets/RaiseHand/vlc-record-2021-12-11-15h28m12s-rtsp___10.10.0.5_28554_fhd-/'
    videos = [video_path + i for i in os.listdir(video_path)]
    videos.sort()

    cap = cv2.VideoCapture(videos[3])
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # half_point = length//7*1
    # cap.set(cv2.CAP_PROP_POS_FRAMES, half_point)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('bbox_joint_01.avi', fourcc, 30.0, (960, 540))
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
                       
        image_array0 = np.copy(frame)
        img_r = cv2.resize(frame, config.get_image_size)[...,::-1]
        img_n = img_r / 255.
        img_n = np.expand_dims(img_n, 0)
        img_n = img_n.transpose(0,3,1,2)
        img_t = torch.from_numpy(img_n).cuda().type(torch.float32)
        
        with torch.no_grad():
            pred = model(img_t)
            
        bboxes, heatmap_hand_tmp, scores, clses = Decoder(pred, (image_array0.shape[0], image_array0.shape[1]))
        image_with_boxes_joint_location_m = np.copy(image_array0)    
        image_with_boxes_joint_location_m = draw_boxes_joint_on_image(image_with_boxes_joint_location_m, bboxes, heatmap_hand_tmp, scores, clses)
        
        image_with_boxes_joint_location_m = cv2.resize(image_with_boxes_joint_location_m, (960, 540))        
        out.write(image_with_boxes_joint_location_m)
        cv2.imshow("detect result", image_with_boxes_joint_location_m)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
#%%
if on_video == False:
    # img = cv2.imread('/home/thomas_yang/ML/datasets/RaiseHand/viveland_pick_by061512v2/images/ \
    #                  vlc-record-2021-11-30-15h25m55s-rtsp___192.168.0.100_8554_fhd-.jpg')
    image_array0 = cv2.imread('/home/thomas_yang/Desktop/416x416.jpg')
    image_with_boxes_joint_location_m = np.copy(image_array0)
    img_r = cv2.resize(image_array0, config.get_image_size)[...,::-1]
    img_n = img_r / 255.
    img_n = np.expand_dims(img_n, 0)
    img_n = img_n.transpose(0,3,1,2)
    img_t = torch.from_numpy(img_n).cuda().type(torch.float32)
    
    pred = model(img_t)
    bboxes, heatmap_hand_tmp, scores, clses = Decoder(pred, (image_array0.shape[0], image_array0.shape[1]))
    # bboxes, scores, clses = Decoder(pred, (img.shape[0], img.shape[1]))
    detect_img = draw_boxes_joint_on_image(image_with_boxes_joint_location_m, bboxes, heatmap_hand_tmp, scores, clses)

    heatmap, reg, wh = torch.split(pred, [config.heads["heatmap"], 
                                          config.heads["reg"], 
                                          config.heads["wh"]], dim=1)    
    heatmap = torch.sigmoid(heatmap)   
    # heatmap = torch.nn.MaxPool2d((7, 7), stride=(1, 1), padding=(3, 3), dilation=1, return_indices=False, ceil_mode=False)(heatmap)
    hm = heatmap[0].cpu().detach().numpy()
    hm = hm.transpose(1, 2, 0)
    hm = cv2.resize(hm, config.get_image_size)
    cv2.imshow('pred[0]', hm[:, :, 0])
    
    detect_img = cv2.resize(detect_img, config.get_image_size)
    cv2.imshow('detect', detect_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




