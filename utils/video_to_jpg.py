#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:43:53 2021

@author: thomas_yang
"""

# import config
import cv2
import os
import numpy as np
import sys

video_path = '/home/thomas_yang/ML/datasets/RaiseHand/vlc-record-2021-12-11-15h28m12s-rtsp___10.10.0.5_28554_fhd-/'
videos = [video_path + i for i in os.listdir(video_path)]
videos.sort()
image_save_dir = '/home/thomas_yang/ML/datasets/RaiseHand/confer_2021_1211_15/images'

for videos_name in videos:
    cap = cv2.VideoCapture(videos_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    count = 0
    while cap.isOpened():        
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        image_name = videos_name.split('/')[-1].split('.')[0] + ('_%08d.jpg'%count)
        cv2.imwrite(os.path.join(image_save_dir, image_name), frame)
        print(image_name)        
        count+=1
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
            
cap.release()
cv2.destroyAllWindows()            