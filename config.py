#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:50:38 2021

@author: thomas_yang
"""

import os

classes_label = {"Not": 0,
                 "Raise": 1,}

finetune = True
finetune_load_epoch = 90

batch_size = 32
epochs = 100
learning_rate = 6e-4

get_image_size = (320, 320)
max_boxes_per_image = 100
downsampling_ratio = 4

num_classes = 2 + 1 
tid_classes = 5
# tid_classes = 55

heads = {"heatmap": num_classes, "wh": 2, "reg": 2, "embed": 256,"tid": tid_classes}
bif = 256

hm_weight = 0.001
off_weight = 1.0
wh_weight = 0.1
reid_eright = 0.1

train_data_dir_1 = '/home/thomas_yang/ML/hTG_RaiseHand/txt_file/'
train_data_list_1 = os.listdir(train_data_dir_1)
train_data_list_1.sort()
train_data_list_1 = [train_data_dir_1 + i for i in train_data_list_1]

train_data_dir_2 = '/home/thomas_yang/ML/hTG_MOT_pytorch/txt_file/'
train_data_list_2 = os.listdir(train_data_dir_2)
train_data_list_2.sort()
train_data_list_2 = [train_data_dir_2 + i for i in train_data_list_2]
train_data_list_2 = train_data_list_2[0:19]

train_data_list = train_data_list_1
# train_data_list = train_data_list_1 + train_data_list_2

top_K = 50
score_threshold = 0.3

