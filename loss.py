#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:25:59 2021

@author: thomas_yang
"""
import config
import torch
import torch.nn as nn

def Loss(pred, gt):
    gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, gt_tid_1d_idx = gt
    heatmap, reg, wh = torch.split(pred, [config.heads["heatmap"], 
                                          config.heads["reg"], 
                                          config.heads["wh"]], dim=1)
    
    # Heatmap Focal Loss
    loss_fcn = nn.BCEWithLogitsLoss()
    gamma=1.5
    alpha=0.25
    reduction = loss_fcn.reduction
    loss_fcn.reduction = 'none'  # required to apply FL to each element

    loss_heatmap = loss_fcn(heatmap, gt_heatmap)
    pred_prob = torch.sigmoid(heatmap)  # prob from logits
    p_t = gt_heatmap * pred_prob + (1 - gt_heatmap) * (1 - pred_prob)
    alpha_factor = gt_heatmap * alpha + (1 - gt_heatmap) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss_heatmap *= alpha_factor * modulating_factor

    loss_heatmap = loss_heatmap.sum()
    # if reduction == 'mean':
    #     loss_heatmap = loss_heatmap.mean()    

    # pred_prob = torch.sigmoid(heatmap)
    # pos_mask = (torch.eq(gt_heatmap, 1.0))
    # neg_mask = (torch.less(gt_heatmap, 1.0))
    # neg_weights = torch.pow(1. - gt_heatmap, 4)
    # # neg_weights = tf.where(K.equal(neg_weights, 1.), 0.00000001, neg_weights)
   
    # pos_loss = torch.log(torch.clip(pred_prob, 1e-5, 1. - 1e-5)) * torch.pow(1. - pred_prob, 2) * pos_mask
    # neg_loss = torch.log(torch.clip(1. - pred_prob, 1e-5, 1. - 1e-5)) * torch.pow(pred_prob, 2.0) * neg_weights * neg_mask

    # num_pos = torch.sum(pos_mask)
    # pos_loss = torch.sum(pos_loss) * 1.
    # neg_loss = torch.sum(neg_loss) * 1.
    
    # loss_heatmap = 0
    # if num_pos == 0:
    #     loss_heatmap = loss_heatmap - neg_loss
    # else:
    #     loss_heatmap = loss_heatmap - (pos_loss + neg_loss) / num_pos
    
    # Offset Loss
    tmp = torch.permute(reg, (0, 2, 3, 1))
    tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
    idx = gt_indices.type(torch.int64)
    idx = idx.unsqueeze(2)
    idx = idx.expand(idx.shape[0], config.max_boxes_per_image, 2)
    tmp = torch.gather(tmp, 1, idx)    
    mask = gt_reg_mask.unsqueeze(2)
    mask = mask.expand(mask.shape[0], config.max_boxes_per_image, 2)
    loss_reg = torch.sum(torch.abs(gt_reg * mask - tmp * mask))
    loss_reg = loss_reg / (torch.sum(mask) + 1e-4)    
 
    # Width/Height Loss
    tmp = torch.permute(wh, (0, 2, 3, 1))
    tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
    tmp = torch.gather(tmp, 1, idx)    
    loss_wh = torch.sum(torch.abs(gt_wh * mask - tmp * mask))
    loss_wh = loss_wh / (torch.sum(mask) + 1e-4)  

    # Total Loss
    loss_heatmap = config.hm_weight * loss_heatmap
    loss_reg = config.off_weight * loss_reg 
    loss_wh = config.wh_weight * loss_wh
    # loss_reid = config.reid_eright * loss_reid
 
    total_loss = (loss_heatmap + 
                  loss_reg + 
                  loss_wh)
    
    return [total_loss, loss_heatmap, loss_reg, loss_wh]

def Loss_reID(pred, gt):
    gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_tid, gt_tid_mask, gt_tid_1d_idx = gt
    heatmap, reg, wh, embed = torch.split(pred[0], [config.heads["heatmap"], 
                                                    config.heads["reg"], 
                                                    config.heads["wh"], 
                                                    config.heads["embed"]], dim=1)
    pred_tid_1d_idx = pred[1]
    
    # Heatmap Focal Loss
    # loss_fcn = nn.BCEWithLogitsLoss()
    # gamma=1.5
    # alpha=0.25
    # reduction = loss_fcn.reduction
    # loss_fcn.reduction = 'none'  # required to apply FL to each element

    # loss_heatmap = loss_fcn(heatmap, gt_heatmap)
    # pred_prob = torch.sigmoid(heatmap)  # prob from logits
    # p_t = gt_heatmap * pred_prob + (1 - gt_heatmap) * (1 - pred_prob)
    # alpha_factor = gt_heatmap * alpha + (1 - gt_heatmap) * (1 - alpha)
    # modulating_factor = (1.0 - p_t) ** gamma
    # loss_heatmap *= alpha_factor * modulating_factor

    # if reduction == 'mean':
    #     loss_heatmap = loss_heatmap.mean()    

    pred_prob = torch.sigmoid(heatmap)
    pos_mask = (torch.eq(gt_heatmap, 1.0))
    neg_mask = (torch.less(gt_heatmap, 1.0))
    neg_weights = torch.pow(1. - gt_heatmap, 4)
    # neg_weights = tf.where(K.equal(neg_weights, 1.), 0.00000001, neg_weights)
   
    pos_loss = torch.log(torch.clip(pred_prob, 1e-5, 1. - 1e-5)) * torch.pow(1. - pred_prob, 2) * pos_mask
    neg_loss = torch.log(torch.clip(1. - pred_prob, 1e-5, 1. - 1e-5)) * torch.pow(pred_prob, 2.0) * neg_weights * neg_mask

    num_pos = torch.sum(pos_mask)
    pos_loss = torch.sum(pos_loss) * 1.
    neg_loss = torch.sum(neg_loss) * 1.
    
    loss_heatmap = 0
    if num_pos == 0:
        loss_heatmap = loss_heatmap - neg_loss
    else:
        loss_heatmap = loss_heatmap - (pos_loss + neg_loss) / num_pos
    
    # return loss  

    
    # Offset Loss
    tmp = torch.permute(reg, (0, 2, 3, 1))
    tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
    idx = gt_indices.type(torch.int64)
    idx = idx.unsqueeze(2)
    idx = idx.expand(idx.shape[0], config.max_boxes_per_image, 2)
    tmp = torch.gather(tmp, 1, idx)    
    mask = gt_reg_mask.unsqueeze(2)
    mask = mask.expand(mask.shape[0], config.max_boxes_per_image, 2)
    loss_reg = torch.sum(torch.abs(gt_reg * mask - tmp * mask))
    loss_reg = loss_reg / (torch.sum(mask) + 1e-4)    
 
    # Width/Height Loss
    tmp = torch.permute(wh, (0, 2, 3, 1))
    tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
    tmp = torch.gather(tmp, 1, idx)    
    loss_wh = torch.sum(torch.abs(gt_wh * mask - tmp * mask))
    loss_wh = loss_wh / (torch.sum(mask) + 1e-4)  

    # reID Loss
    mask = gt_tid_mask.unsqueeze(2)
    mask = mask.expand(mask.shape[0], config.max_boxes_per_image, config.tid_classes)
    loss_reid = torch.sum(torch.abs(gt_tid * mask - pred_tid_1d_idx * mask)**2)

    # Total Loss
    loss_heatmap = config.hm_weight * loss_heatmap
    loss_reg = config.off_weight * loss_reg 
    loss_wh = config.wh_weight * loss_wh
    # loss_reid = config.reid_eright * loss_reid
 
    total_loss = (loss_heatmap + 
                  loss_reg + 
                  loss_wh+
                  loss_reid)
    
    return [total_loss, loss_heatmap, loss_reg, loss_wh, loss_reid]












