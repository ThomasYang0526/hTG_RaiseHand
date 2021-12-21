#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:40:33 2021

@author: thomas_yang
"""

import cv2

txt_path = '/home/thomas_yang/ML/hTG_RaiseHand/txt_file/'
txt_file = 'office_2021_1129_14.txt'
circle_color = [(0, 255, 0), (0, 0, 255)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']

with open(txt_path + txt_file, 'r') as f:
    for line_idx, line in enumerate(f.readlines()):
        line_split = line.split()
        img_name = line_split[0]

        img = cv2.imread(img_name)
        for i in range(len(line_split[3:])//6):
            xmin = int(float(line_split[3 + i*6 + 0]))
            ymin = int(float(line_split[3 + i*6 + 1]))
            xmax = int(float(line_split[3 + i*6 + 2]))
            ymax = int(float(line_split[3 + i*6 + 3]))
            class_id = int(float(line_split[3 + i*6 + 4]))
            tracking_id = int(float(line_split[3 + i*6 + 5]))
            if class_id >= 0:
                x_c = (xmin + xmax)//2
                y_c = (ymin + ymax)//2
                cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=circle_color[class_id], thickness=2)
                cv2.circle(img, (x_c, y_c), 13, (0, 0, 255), -1)

        img = cv2.resize(img, (512, 512))
        cv2.imshow('img', img)
        print(line_idx)
        if cv2.waitKey(1) == ord('q'):
            break        

cv2.destroyAllWindows()        
        