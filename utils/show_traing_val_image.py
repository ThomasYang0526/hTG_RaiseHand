#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:28:56 2021

@author: thomas_yang
"""

import config
import numpy as np
import cv2
import torch

def show_traing_val_image(images, gt_heatmap, pred, training = False):
    
    images = images.cpu().numpy().transpose((0,2,3,1))    
    gt_heatmap = gt_heatmap.cpu().numpy().transpose((0,2,3,1))
    
    pred = torch.sigmoid(pred)  # prob from logits
    pred = pred.cpu().detach().numpy().transpose((0,2,3,1))

    name_dict = {False:['val_img', 'val_hms'], True:['train_img', 'train_hms']}
    
    min_show_figures = images.shape[0] if images.shape[0]%config.batch_size == 0 else images.shape[0]%config.batch_size       

    imgs = np.concatenate((images[0:min(min_show_figures,8),...,::-1]*255).astype(np.uint8), axis = 1)
    imgs = cv2.resize(imgs, (imgs.shape[1]//4, imgs.shape[0]//4))
    
    gt_pre_heatmap = \
    np.concatenate((np.concatenate(np.max(gt_heatmap[0:min(min_show_figures,8), :, :, 0:config.num_classes], axis=-1), axis = 1),
                    np.concatenate(np.max(      pred[0:min(min_show_figures,8), :, :, 0:config.num_classes], axis=-1), axis = 1)), axis = 0)   
    
    hms = gt_pre_heatmap 
    
    cv2.imshow(name_dict[training][0], imgs)
    cv2.imshow(name_dict[training][1], hms)
    cv2.waitKey(1)