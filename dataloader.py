#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:44:13 2021

@author: thomas_yang
"""

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import config
from utils.gaussian import gaussian_radius, draw_umich_gaussian
import imgaug.augmenters as iaa

class Mydata(Dataset):
    def __init__(self, txt_path):        
        self.info = open(txt_path, encoding='utf8').read().split('\n')[:-1]
        self.image_names = [i.split(' ')[0] for i in self.info]
        self.image_labels = [i for i in self.info]
        self.input_image_height = config.get_image_size[0]
        self.input_image_width = config.get_image_size[1]   
        self.max_boxes_per_image = config.max_boxes_per_image            
        self.downsampling_ratio = config.downsampling_ratio # efficientdet: 8, others: 4
        self.features_shape = np.array(config.get_image_size, dtype=np.int32) // self.downsampling_ratio # efficientnet: 64*64        
        self.downsampling_ratio = config.downsampling_ratio # efficientdet: 8, others: 4
        self.features_shape = np.array(config.get_image_size, dtype=np.int32) // self.downsampling_ratio # efficientnet: 64*64
        self.seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0, 1)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255)),
            iaa.AddToHueAndSaturation((-10, 10), per_channel=True),
            iaa.GammaContrast((0.75, 1.25)),
            iaa.MultiplyHueAndSaturation(),
            iaa.MotionBlur(k=3),
        ], random_order=True) 
    
    def __getitem__(self, idx):
        image, boxes = self.__get_image_information(self.image_labels[idx]) 
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx = self.get_gt_values(boxes)
        image = np.transpose(image, axes=[2, 0, 1]).astype(np.float32) / 255
        gt_heatmap = np.transpose(gt_heatmap, axes=[2, 0, 1]).astype(np.float32)
        return image, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx
        
    def __len__(self):
        return len(self.image_names)      

    def __get_image_information(self, labels_info):
        """
        boxes_array: numpy array, shape = (max_boxes_per_image, 6(xmin, ymin, xmax, ymax, class_id, tid))
        """
        line_list = labels_info.split(' ')
        txt_interval = 6        
        image_file, image_height, image_width = line_list[:3]
        image_height, image_width = int(float(image_height)), int(float(image_width))
        boxes = []
        num_of_boxes = (len(line_list) - 3) / txt_interval
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be type 'int'.")
            
        for index in range(num_of_boxes):
            if index < self.max_boxes_per_image:
                xmin = int(float(line_list[3 + index * txt_interval]))
                ymin = int(float(line_list[3 + index * txt_interval + 1]))
                xmax = int(float(line_list[3 + index * txt_interval + 2]))
                ymax = int(float(line_list[3 + index * txt_interval + 3]))
                class_id = int(line_list[3 + index * txt_interval + 4])
                tid = int(line_list[3 + index * txt_interval + 5])
                xmin, ymin, xmax, ymax = self.box_preprocess(image_height, image_width, xmin, ymin, xmax, ymax)
                boxes.append([xmin, ymin, xmax, ymax, class_id, tid])
                
        num_padding_boxes = self.max_boxes_per_image - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1, -1])
        boxes_array = np.array(boxes, dtype=np.float32)

        image = cv2.imread(image_file)[...,::-1]
        image = cv2.resize(image, config.get_image_size)
        # seq = iaa.Sometimes(0.5, self.seq)
        # image = seq(image=image)
        
        return image, boxes_array 

    def box_preprocess(self, h, w, xmin, ymin, xmax, ymax):
        resize_ratio = [self.input_image_height / h, self.input_image_width / w]
        xmin = int(resize_ratio[1] * xmin)
        xmax = int(resize_ratio[1] * xmax)
        ymin = int(resize_ratio[0] * ymin)
        ymax = int(resize_ratio[0] * ymax)
        return xmin, ymin, xmax, ymax

    def get_gt_values(self, boxes):        
        boxes = boxes[boxes[:, 4] != -1]        
        label_class = boxes[boxes[:, 4] >= 0]
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx = self.__decode_label(label_class)                        
        return gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx

    def __decode_label(self, label):
        hm = np.zeros(shape=(self.features_shape[0], self.features_shape[1], config.num_classes), dtype=np.float32)
        reg = np.zeros(shape=(config.max_boxes_per_image, 2), dtype=np.float32)
        wh = np.zeros(shape=(config.max_boxes_per_image, 2), dtype=np.float32)
        reg_mask = np.zeros(shape=(config.max_boxes_per_image), dtype=np.float32)
        ind = np.zeros(shape=(config.max_boxes_per_image), dtype=np.float32)
        
        tid = np.zeros(shape=(config.max_boxes_per_image, config.tid_classes), dtype=np.float32)        
        tid_mask = np.zeros(shape=(config.max_boxes_per_image), dtype=np.float32)
        tid_1d_idx = np.zeros(shape=(config.max_boxes_per_image), dtype=np.float32)
        
        for j, item in enumerate(label): #原始座標
            item_down = item[:4] / self.downsampling_ratio #原始座標/縮小比例(8)
            xmin, ymin, xmax, ymax = item_down            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(self.features_shape[1]-1, xmax)
            ymax = min(self.features_shape[0]-1, ymax)
            
            class_id = item[4].astype(np.int32)
            h, w = ymax - ymin, xmax - xmin
            radius = gaussian_radius((int(h), int(w)))
            radius = max(0, int(radius))   

            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = np.array([ctr_x, ctr_y], dtype=np.float32)
            center_point_int = center_point.astype(np.int32)
                        
            draw_umich_gaussian(hm[:, :, 0], center_point_int, radius)
            hm[center_point_int[1], center_point_int[0], class_id + 1] = 1.0
            
            reg[j] = center_point - center_point_int
            
            wh[j] = 1. * w, 1. * h
            reg_mask[j] = 1
            ind[j] = center_point_int[1] * self.features_shape[1] + center_point_int[0]
            
            # if item[5] != -1: 
            #     # tmp = np.zeros((1, config.tid_classes))
            #     # tmp[0][[int(item[5])]] = 1.0
            #     tid[j][[int(item[5])]] = 1.0
            #     tid_mask[j] = 1
            #     tid_1d_idx[j] = center_point_int[1] * self.features_shape[1] + center_point_int[0]
                
        return hm, reg, wh, reg_mask, ind, tid, tid_mask, tid_1d_idx
        
if __name__ == '__main__':
    path = '/home/thomas_yang/ML/hTC_MOT/txt_file/pose_detection/ethz_01.txt'    
    dataset = Mydata(path)
    dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=0)
    for i in dataloader:
        image, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask = i
        print(i[0].shape, i[1].shape, i[2].shape, i[3].shape, i[4].shape, i[5].shape, i[6].shape, i[7].shape)
    

