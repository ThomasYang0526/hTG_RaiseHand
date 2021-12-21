#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:42:38 2021

@author: thomas_yang
"""

import config
import numpy as np
import torch

def Decoder(pred, original_image_size):
        original_image_size = np.array(original_image_size, dtype=np.float32)
        input_image_size = np.array(config.get_image_size, dtype=np.float32)
        downsampling_ratio = config.downsampling_ratio
        score_threshold = config.score_threshold        
        
        heatmap_, reg, wh = torch.split(pred, [config.heads["heatmap"], 
                                               config.heads["reg"], 
                                               config.heads["wh"]], dim=1)
        
        # __topK
        # batch_size = heatmap.shape[0]
        heatmap_hand = heatmap_[:, 1:3, :, :]
        heatmap = heatmap_[:, 0:1, :, :]
        heatmap = torch.sigmoid(heatmap)        
        heatmap2 = torch.nn.MaxPool2d((7, 7), stride=(1, 1), padding=(3, 3), dilation=1, return_indices=False, ceil_mode=False)(heatmap)
        keep = torch.eq(heatmap, heatmap2).type(torch.float32)
        heatmap_nms = keep*heatmap2
        heatmap_nms_tmp = torch.permute(heatmap_nms, (0, 2, 3, 1))
        B, H, W, C = heatmap_nms_tmp.shape
        heatmap_nms_tmp = torch.reshape(heatmap_nms_tmp, (B, -1)) 
        
        topk_scores, topk_inds = torch.topk(heatmap_nms_tmp, config.top_K, dim=1)
        topk_clses = topk_inds % C
        topk_xs = (torch.div(topk_inds, C, rounding_mode='floor') % W).type(torch.float32)
        topk_ys = (torch.div(torch.div(topk_inds, C, rounding_mode='floor'), W, rounding_mode='floor')).type(torch.float32)
        topk_inds = (topk_ys * W + topk_xs).type(torch.int32)
        scores, inds, clses, ys, xs = topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
        
        tmp = torch.permute(reg, (0, 2, 3, 1))
        tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
        idx = inds.type(torch.int64)
        idx = idx.unsqueeze(2)
        idx = idx.expand(idx.shape[0], config.top_K, 2)
        reg_tmp = torch.gather(tmp, 1, idx)
        

        xs = torch.reshape(xs, shape=(B, config.top_K, 1)) + reg_tmp[:, :, 0:1]
        ys = torch.reshape(ys, shape=(B, config.top_K, 1)) + reg_tmp[:, :, 1:2]

        tmp = torch.permute(wh, (0, 2, 3, 1))
        tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
        wh_tmp = torch.gather(tmp, 1, idx)

        tmp = torch.permute(heatmap_hand, (0, 2, 3, 1))
        tmp = torch.reshape(tmp, shape=(tmp.shape[0], -1, tmp.shape[-1]))
        heatmap_hand_tmp = torch.gather(tmp, 1, idx)

        clses = torch.reshape(clses, (B, config.top_K, 1)).type(torch.float32)
        scores = torch.reshape(scores, (B, config.top_K, 1))        
        bboxes = torch.cat([xs - wh_tmp[..., 0:1] / 2,
                            ys - wh_tmp[..., 1:2] / 2,
                            xs + wh_tmp[..., 0:1] / 2,
                            ys + wh_tmp[..., 1:2] / 2], dim=2)
        
        # detections = torch.cat([bboxes, scores, clses, embed_tmp], dim=2)
        resize_ratio = original_image_size / input_image_size
        bboxes[:, :, 0::2] = bboxes[:, :, 0::2] * downsampling_ratio * resize_ratio[1]
        bboxes[:, :, 1::2] = bboxes[:, :, 1::2] * downsampling_ratio * resize_ratio[0]        
        bboxes[:, :, 0::2] = torch.clip(bboxes[:, :, 0::2], 0, original_image_size[1])
        bboxes[:, :, 1::2] = torch.clip(bboxes[:, :, 1::2], 0, original_image_size[0])
        
        score_mask = scores >= score_threshold        
        # print(score_mask)
        # bboxes = bboxes[score_mask.expand(score_mask.shape[0], score_mask.shape[1], bboxes.shape[-1])].reshape(-1, bboxes.shape[-1])
        # scores = scores[score_mask.expand(score_mask.shape[0], score_mask.shape[1], scores.shape[-1])].reshape(-1, scores.shape[-1])
        # clses = clses[score_mask.expand(score_mask.shape[0], score_mask.shape[1], clses.shape[-1])].reshape(-1, clses.shape[-1])
        # embed_tmp = embed_tmp[score_mask.expand(score_mask.shape[0], score_mask.shape[1], embed_tmp.shape[-1])].reshape(-1, embed_tmp.shape[-1])
        bboxes = bboxes[score_mask.expand(score_mask.shape[0], score_mask.shape[1], bboxes.shape[-1])].reshape(-1, bboxes.shape[-1]).cpu().detach().numpy()
        heatmap_hand_tmp = heatmap_hand_tmp[score_mask.expand(score_mask.shape[0], score_mask.shape[1], heatmap_hand_tmp.shape[-1])].reshape(-1, heatmap_hand_tmp.shape[-1]).cpu().detach().numpy()
        scores = scores[score_mask.expand(score_mask.shape[0], score_mask.shape[1], scores.shape[-1])].reshape(-1, scores.shape[-1]).cpu().detach().numpy().reshape(-1)
        clses = clses[score_mask.expand(score_mask.shape[0], score_mask.shape[1], clses.shape[-1])].reshape(-1, clses.shape[-1]).cpu().detach().numpy().reshape(-1)
        # embed_tmp = embed_tmp[score_mask.expand(score_mask.shape[0], score_mask.shape[1], embed_tmp.shape[-1])].reshape(-1, embed_tmp.shape[-1]).cpu().detach().numpy()        
        
        return [bboxes, heatmap_hand_tmp, scores, clses]

