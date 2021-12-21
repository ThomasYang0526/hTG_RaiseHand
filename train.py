#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:26:19 2021

@author: thomas_yang
"""

import config
import torch
import time
import numpy as np
from dataloader import Mydata
from torch.utils.data import DataLoader
from resnetFPN import MyResNet50
from loss import Loss
from torch.utils.tensorboard import SummaryWriter
from utils.show_traing_val_image import show_traing_val_image
from torch.optim.lr_scheduler import StepLR
import datetime

device = torch.device('cuda')
dataset = Mydata(config.train_data_list[0])
for txt_file in config.train_data_list[1:]:
    dataset += Mydata(txt_file)

dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=0, shuffle=True)

epoch_steps = len(dataset)//config.batch_size
model = MyResNet50().cuda()

load_weights_from_epoch = config.finetune_load_epoch
if config.finetune:
    model.load_state_dict(torch.load('./saved_model/resnet50FPN_256_epoch_{}.pth'.format(load_weights_from_epoch)), strict = False)
else:
    load_weights_from_epoch = -1

optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = StepLR(optim, epoch_steps, gamma=0.96)
writer = SummaryWriter('logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

for epoch in range(load_weights_from_epoch + 1, config.epochs):    
    for step, train_data in enumerate(dataloader):

        ST = time.time()
        train_data = [i.to(device) for i in train_data]
        images, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, tid_1d_idx = train_data     
        pred = model(images)
        loss = Loss(pred, train_data[1:])
        optim.zero_grad()
        loss[0].backward()
        optim.step()
        scheduler.step()
        ET = time.time()  
        
        global_steps = epoch*epoch_steps+ step
        writer.add_scalar('Loss/Total Loss', loss[0].item(), global_steps)
        writer.add_scalar('Loss/Heatmap Loss', loss[1].item(), global_steps)
        writer.add_scalar('Loss/Offset Loss', loss[2].item(), global_steps)
        writer.add_scalar('Loss/WH Loss', loss[3].item(), global_steps)
        # writer.add_scalar('Loss/ID Loss', loss[4].item(), global_steps)
        writer.add_scalar('LR/learning rate', scheduler.get_last_lr()[0], global_steps)
                
        # writer.add_images('Image/Gt', images[0:4, :, :, :], global_steps)
        # writer.add_images('Heatmap/Gt', gt_heatmap[0:4, :, :, :], global_steps)
        # writer.add_images('Heatmap/Pre', torch.sigmoid(pred[0][0:4, 0:1, :, :]), global_steps)
        
        show_traing_val_image(images, gt_heatmap, pred, training = True)
        print('epoch: {}, step: {}/{}, Total loss {:.3f}, Heatmap {:.3f}, Offset {:.3f}, WH {:.3f}, Time {:.3f}'.format(epoch, step, epoch_steps, 
                                                                                                                        loss[0].item(),
                                                                                                                        loss[1].item(),
                                                                                                                        loss[2].item(),
                                                                                                                        loss[3].item(),                                                                                                                                    
                                                                                                                        ET-ST,))
    torch.save(model.state_dict(), './saved_model/resnet50FPN_256_epoch_{}.pth'.format(epoch))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        