#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:14:22 2021

@author: thomas_yang
"""

import os
import cv2

save_txt_dir = '/home/thomas_yang/ML/hTG_RaiseHand/txt_file/KH5G_2.txt'
load_txt_dir = '/home/thomas_yang/ML/datasets/RaiseHand/KH5G/2021-09-24-kaohsiung/train/labels_with_ids/'
txt_files = [load_txt_dir + i for i in os.listdir(load_txt_dir)]
txt_files.sort()

imgs_info = []
for idx, txt_fname in enumerate(txt_files):
    print('{:d}------{:d}'.format(idx+1, len(txt_files)))
    img_fname = txt_fname.replace('.txt', '.jpg').replace('labels_with_ids', 'images')
    if not os.path.isfile(img_fname):
        img_fname = txt_fname.replace('.txt', '.png').replace('labels_with_ids', 'images')
        
    img = cv2.imread(img_fname)
    img_height, img_width = img.shape[:2]    
    
    with open(txt_fname, 'r') as f:
        img_info = [img_fname, str(img_height), str(img_width)]
        for line_idx, line in enumerate(f.readlines()):
            line = line.strip().split()
            clases = int(line[0])
            tid_curr = int(line[1])
            xmin = float(line[2]) * img_width
            ymin = float(line[3]) * img_height
            xmax = float(line[4]) * img_width
            ymax = float(line[5]) * img_height
            # xmin = xcen - w/2
            # ymin = ycen - h/2
            # xmax = xcen + w/2
            # ymax = ycen + h/2
            label_str = '{:.2f} {:.2f} {:.2f} {:.2f} {:d} {:d}'.format(xmin, ymin, xmax, ymax, clases, tid_curr)
            img_info.append(label_str)
    img_info = ' '.join(img_info)+'\n'
    imgs_info.append(img_info)
    

with open(save_txt_dir, 'w') as f:
    for label_str in imgs_info:
        f.write(label_str)