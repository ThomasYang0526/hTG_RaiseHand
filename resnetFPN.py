#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:51:14 2021

@author: thomas_yang
"""

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import math

import config
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import MobileNetV2
from torch.nn import Conv2d, Upsample, BatchNorm2d, ReLU, Softmax

class MyResNet50(ResNet):
    def __init__(self):
        super(MyResNet50, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.conv_up1 = Conv2d( 64, config.bif, 1)
        self.conv_up2 = Conv2d(128, config.bif, 1)
        self.conv_up3 = Conv2d(256, config.bif, 1)
        self.conv_up4 = Conv2d(512, config.bif, 1)
        self.Upsample = Upsample(scale_factor=2)
        
        self.conv_heatmap_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_heatmap = BatchNorm2d(config.bif)
        self.relu_heatmap = ReLU()
        self.conv_heatmap_2 = Conv2d(config.bif, config.heads["heatmap"], 1, padding=(0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_reg_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_reg = BatchNorm2d(config.bif)
        self.relu_reg = ReLU()
        self.conv_reg_2 = Conv2d(config.bif, config.heads["reg"], 1, padding=(0, 0), bias=False)

        self.conv_wh_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
        self.bn_wh = BatchNorm2d(config.bif)
        self.relu_wh = ReLU()
        self.conv_wh_2 = Conv2d(config.bif, config.heads["wh"], 1, padding=(0, 0), bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        p2_output = self.conv_up1(x1)
        p3_output = self.conv_up2(x2)
        p4_output = self.conv_up3(x3)
        p5_output = self.conv_up4(x4)
               
        p4_output = p4_output + self.Upsample(p5_output)
        p3_output = p3_output + self.Upsample(p4_output)
        p2_output = p2_output + self.Upsample(p3_output)

        heatmap = self.conv_heatmap_1(p2_output)
        heatmap = self.bn_heatmap(heatmap)
        heatmap = self.relu_heatmap(heatmap)
        heatmap = self.conv_heatmap_2(heatmap)

        reg = self.conv_reg_1(p2_output)
        reg = self.bn_reg(reg)
        reg = self.relu_reg(reg)
        reg = self.conv_reg_2(reg)        

        wh = self.conv_wh_1(p2_output)
        wh = self.bn_wh(wh)
        wh = self.relu_wh(wh)
        wh = self.conv_wh_2(wh)   
        
        return torch.cat([heatmap, reg, wh], dim=1)

# class MyResNet50(MobileNetV2):
#     def __init__(self):
#         super(MyResNet50, self).__init__(width_mult = 2.0)
#         self.conv_up1 = Conv2d( 48, config.bif, 1)
#         self.conv_up2 = Conv2d( 64, config.bif, 1)
#         self.conv_up3 = Conv2d(192, config.bif, 1)
#         self.conv_up4 = Conv2d(640, config.bif, 1)
#         self.Upsample = Upsample(scale_factor=2)
        
#         self.conv_heatmap_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
#         self.bn_heatmap = BatchNorm2d(config.bif)
#         self.relu_heatmap = ReLU()
#         self.conv_heatmap_2 = Conv2d(config.bif, config.heads["heatmap"], 1, padding=(0, 0), bias=False)
#         self.sigmoid = nn.Sigmoid()

#         self.conv_reg_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
#         self.bn_reg = BatchNorm2d(config.bif)
#         self.relu_reg = ReLU()
#         self.conv_reg_2 = Conv2d(config.bif, config.heads["reg"], 1, padding=(0, 0), bias=False)

#         self.conv_wh_1 = Conv2d(config.bif, config.bif, 3, padding=(1, 1), bias=False)
#         self.bn_wh = BatchNorm2d(config.bif)
#         self.relu_wh = ReLU()
#         self.conv_wh_2 = Conv2d(config.bif, config.heads["wh"], 1, padding=(0, 0), bias=False)
        
#     def forward(self, x):
#         x0 = self.feature_00(x)
#         x1 = self.feature_01(x0)
#         x2 = self.feature_02(x1)
#         x3 = self.feature_03(x2)
#         x4 = self.feature_04(x3)
#         x5 = self.feature_05(x4)
#         x6 = self.feature_06(x5)
#         x7 = self.feature_07(x6)
#         x8 = self.feature_08(x7)
#         x9 = self.feature_09(x8)
#         x10 = self.feature_10(x9)
#         x11 = self.feature_11(x10)
#         x12 = self.feature_12(x11)
#         x13 = self.feature_13(x12)
#         x14 = self.feature_14(x13)
#         x15 = self.feature_15(x14)
#         x16 = self.feature_16(x15)
#         x17 = self.feature_17(x16)
#         # x18 = self.feature_18(x17)
        
#         p2_output = self.conv_up1(x3)
#         p3_output = self.conv_up2(x6)
#         p4_output = self.conv_up3(x13)
#         p5_output = self.conv_up4(x17)
               
#         p4_output = p4_output + self.Upsample(p5_output)
#         p3_output = p3_output + self.Upsample(p4_output)
#         p2_output = p2_output + self.Upsample(p3_output)

#         heatmap = self.conv_heatmap_1(p2_output)
#         heatmap = self.bn_heatmap(heatmap)
#         heatmap = self.relu_heatmap(heatmap)
#         heatmap = self.conv_heatmap_2(heatmap)

#         reg = self.conv_reg_1(p2_output)
#         reg = self.bn_reg(reg)
#         reg = self.relu_reg(reg)
#         reg = self.conv_reg_2(reg)        

#         wh = self.conv_wh_1(p2_output)
#         wh = self.bn_wh(wh)
#         wh = self.relu_wh(wh)
#         wh = self.conv_wh_2(wh)          
        
#         return torch.cat([heatmap, reg, wh], dim=1)

if __name__ == '__main__':
 
    from dataloader import Mydata
    from torch.utils.data import DataLoader
   
    device = torch.device('cuda')
    path = '/home/thomas_yang/ML/hTG_RaiseHand/txt_file/viveland.txt'    
    dataset = Mydata(path)    
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    
    for step, train_data in enumerate(dataloader):
        train_data = [i.to(device) for i in train_data]
        image, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx = train_data
        if step == 0:
            break
    
    # x = torch.ones((1,3,416,416)).cuda()
    # tid = torch.ones((1, 100)).cuda()
    model = MyResNet50().cuda().eval()
    y = model(image)

    dummy_input = torch.randn(1, 3, 416, 416, device="cuda")
    input_names = [ "input1"]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
