#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:04 2023

@author: chegde
"""
"""
Test localization on April 2022 dataset
"""

from get_pos_ori import get_pos_ori
from datetime import datetime
from transform import Transforms, Plotting, Utils
%matplotlib qt
import matplotlib.pyplot as plt
from ground_truth import April2022_GT, position_rescale

start_time = '20220422_172939' # yyyymmdd_HHMMSS
end_time =  '20220422_175252' # yyyymmdd_HHMMSS

# start_time = '20221118_171415' # Nov dataset
# end_time = '20221118_174825'

start_time = datetime.strptime(start_time, '%Y%m%d_%H%M%S')
end_time = datetime.strptime(end_time, '%Y%m%d_%H%M%S')

positions, orientations, timestamps = get_pos_ori(start_time, end_time)
positions_rescaled = position_rescale(positions)


# Get positions and orientations of interest
gt_timestamps = ['20220422_172942', '20220422_173420', '20220422_173440', 
                 '20220422_174750', '20220422_174820', '20220422_174910', 
                 '20220422_175110', '20220422_175200', '20220422_175250']
for i in range(len(gt_timestamps)):
    gt_timestamps[i] = datetime.strptime(gt_timestamps[i], '%Y%m%d_%H%M%S')

i = 0
positions_test = []
orientations_test = []
for ts in timestamps:
    if ts in gt_timestamps:
        pos = []
        ori = []
        for j in range(len(positions_rescaled[i])):
            if positions_rescaled[i][j][0] <= 12:
                pos.append(positions_rescaled[i][j])
                ori.append(orientations[i][j])
        positions_test.append(pos)
        orientations_test.append(ori)
    i += 1

#### Load ground truth ####
positions_gt, orientations_gt = April2022_GT()

#### Plot positions and orientations ####
P = Plotting()
floor = plt.imread('ep6_floorplan_measured_half_gridded_1_meter.jpg')
index = 2
fig, ax = plt.subplots()
P.plot_pos_ori_rescaled(orientations_test[index], positions_test[index], floor, [fig,ax], color='r')
P.plot_pos_ori_rescaled(orientations_gt[index], positions_gt[index], floor, [fig,ax], color='g')
#fig.savefig('posori.png')
orientations_test[index] = [ int(x) for x in orientations_test[index] ]

P = Plotting()
floor = plt.imread('ep6_floorplan_measured_half_gridded_1_meter.jpg')
index = 1391
fig, ax = plt.subplots()
P.plot_hyeok(orientations[index], positions[index], floor, [fig,ax], color='r')