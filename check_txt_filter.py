#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:20:50 2021

@author: thomas_yang
"""


import cv2
import numpy as np


txt_path = '/home/thomas_yang/ML/hTG_RaiseHand/txt_file/'
txt_file = 'confer_2021_1211_15.txt'
circle_color = [(0, 255, 0), (0, 0, 255)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']

img_list = []
with open(txt_path + txt_file, 'r') as f:
    for line_idx, line in enumerate(f.readlines()):
        img_list.append(line)

idx = 0
remove_list = []
while True:
    line = img_list[idx]
    line_split = line.split()
    img_name = line_split[0]

    img = cv2.imread(img_name)
    cv2.putText(img, str(idx), (50, 50), 0, 2, (0, 0, 255), 2)
    if idx in remove_list:
        img[:, 1500:, :] = 255
    
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

    img = cv2.resize(img, (960, 540))
    cv2.imshow('img', img)
    print(idx)
    if cv2.waitKey(0) == ord('q'):
        break
    elif cv2.waitKey(0) == ord('2'):
        idx+=1
    elif cv2.waitKey(0) == ord('5'):
        idx-=1
    elif cv2.waitKey(0) == ord('1'):
        if idx not in remove_list:
            remove_list.append(idx)
            remove_list.sort()
            print(remove_list)
            print('Remove this image, Total remove frames', len(remove_list))
            idx += 1
        else:
            print('Already Removeq')
            idx += 1
    elif cv2.waitKey(0) == ord('3'):
        if idx in remove_list:
            remove_list.remove(idx)
            remove_list.sort()
            idx += 1
        else:
            print('No item')

cv2.destroyAllWindows()   


#%% save new txt

# new_img_list = []
# with open(txt_path + txt_file, 'r') as f:
#     for line_idx, line in enumerate(f.readlines()):
#         if line_idx not in remove_list:
#             new_img_list.append(line)
            
# new_txt_file = 'confer_2021_1211_15_.txt'            
# save_txt_dir =  txt_path + new_txt_file           
# with open(save_txt_dir, 'w') as f:
#     for label_str in new_img_list:
#         f.write(label_str)






















  
       