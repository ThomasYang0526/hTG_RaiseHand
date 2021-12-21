#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:07:59 2021

@author: thomas_yang
"""

import os
import csv
import collections

info_list = []
with open('/home/thomas_yang/ML/datasets/RaiseHand/KH5G/2021-09-24-kaohsiung/train/label.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        info_list.append(row)
        print(row)
      
info_list.sort()
# print(info_list)

dicts = collections.defaultdict(list)
for i in info_list[1:]:
    if 'Thomas' in i[-1]:
        tid = ['1']
    elif 'Jason' in i[-1]:
        tid = ['2']
    elif 'Ziv' in i[-1]:
        tid = ['3']
    else:
        tid = ['-1']
    
    if 'not' in i[-2]:
        clses = ['0']
    else:
        clses = ['1']
    
    dicts[i[0] + '.txt'].append(clses+ tid+ [i[4][0:8]]+ [i[6][0:8]]+ [i[5][0:8]]+ [i[7][0:8]])


path = '/home/thomas_yang/ML/datasets/RaiseHand/KH5G/2021-09-24-kaohsiung/train/labels_with_ids'
for i in dicts:
    with open(os.path.join(path, i), 'w') as f:
        for j in dicts[i]:
            f.write(' '.join(j) +'\n')    
    