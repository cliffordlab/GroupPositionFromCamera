#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:27:33 2023

@author: chegde
"""

import sys
sys.path.append('localization')
from localization.get_pos_ori import get_pos_ori
from dataset_preparation import PrepareDatasets
from clustering import dbscan_clusters
from cluster_performance_metrics import cluster_metrics, room_level_testing, metrics_vs_D
from datetime import datetime
%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
import math


floorplan_path = 'ep6_floorplan_measured_half_gridded_1_meter.jpg' # Load floorplan image
datasets = PrepareDatasets() 
    
# If you have access to raw data from camera systems, set use_sample_data=False
# if not, use sample data provided by setting use_sample_data=True
use_sample_data = True
#%%%%%%%%%%%%%%%%%%%%%%%%% Load data %%%%%%%%%%%%%%%%%%%%%%#

#### Obtain ground truth ####
gt_groups = datasets.Nov2022()[1]
gt = datasets.Nov2022_clusters()
gt_timestamps = gt[0]
gt_centroids = gt[1]
    
if use_sample_data == False:
    
    # Estimate positions and orientations 
    start_time = '20221118_171420'
    end_time = '20221118_174820'
        
    start_time = datetime.strptime(start_time, '%Y%m%d_%H%M%S')
    end_time = datetime.strptime(end_time, '%Y%m%d_%H%M%S')
    positions, orientations, timestamps = get_pos_ori(start_time, end_time)

else:
    # Load sample positions and orientations
    with open('sample_data/positions_samples', 'rb') as fp:
        positions = pickle.load(fp)
    with open('sample_data/orientations_samples', 'rb') as fp:
        orientations = pickle.load(fp)
    with open('sample_data/timestamps_samples', 'rb') as fp:
        timestamps = pickle.load(fp)

#%%%%%%%%%% Preprocess positions and orientations to remove FP %%%%%%%%%%#
# Remove people who's positions and ori don't change over S seconds
def find_index(arr_list, target_arr):
    for i in range(len(arr_list)):
        if (arr_list[i] == target_arr).all():
            return i

S = 10
for i in range(len(positions)-S):
    len2 = len(positions[i])
    j=0
    while j < len2:
        pos_start = positions[i][j]
        ori_start = orientations[i][j]
        fp_flag = 0
        fp_indices = []
        for s in range(S):
            if np.any(np.all(pos_start==positions[i+s+1], axis=1)):
                ind = find_index(positions[i+s+1], pos_start)
                if ori_start == orientations[i+s+1][ind]:
                    fp_flag += 1
                    fp_indices.append(ind)
        if fp_flag == S:
            positions[i].pop(j)
            orientations[i].pop(j)
            for k in range(S):
                positions[i+k+1].pop(fp_indices[k])
                orientations[i+k+1].pop(fp_indices[k])
        len2 = len(positions[i])
        j += 1
  
# Get common timestamps between estimated and ground truth 
common_timestamps = sorted(list(set(timestamps).intersection(gt_timestamps)))

#%%%%%%%%%%%% Perform group identification and localization %%%%%%%%%%%%%#
dbscan_centroids = []
dbscan_groups = []
true_times = []
false_times = []
positions_common = []
orientations_common = []
count = 0
for p, o, t in zip(positions, orientations, timestamps):
    print('Instance ', count, ' of ', len(positions))
    count += 1
    if t in common_timestamps:
        eps = 2.1*17.5 # 2.1m isconsidered for eps. 17.5 pixels=1m
        groups, centroids = dbscan_clusters(p, o, eps=eps, only_positions=False)
        dbscan_centroids.append(centroids)
        dbscan_groups.append(groups)
        true_times.append(t)
        positions_common.append(p)
        orientations_common.append(o)
    else:
        false_times.append(t)
        
# Get GT groups that are part of common timestamps
gt_groups_common = []
gt_centroids_common = []
for g, c, t in zip(gt_groups, gt_centroids, gt_timestamps):
    if t in common_timestamps:
        gt_groups_common.append(g)
        gt_centroids_common.append(c)
        
# # Remove dbscan centroids with no people around
# dbscan_centroids_clean = []
# for i in range(len(dbscan_centroids)):
#     pos = positions_common[i]
#     cent = dbscan_centroids[i]
#     clean_cent = []
#     for j in range(len(cent)):
#         num_neighbors = 0
#         for k in range(len(pos)):
#             d = np.sqrt((pos[k][0]-cent[j][0])**2 + (pos[k][1]-cent[j][1])**2)
#             if d <= 35:
#                 num_neighbors += 1
#         if num_neighbors > 0:
#             clean_cent.append(cent[j])
#     dbscan_centroids_clean.append(clean_cent)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#   
TPR, precision, recall, F1, MAE = cluster_metrics(detected_centroids=dbscan_centroids_clean[3::], 
                                                  true_centroids=gt_centroids_common[3::], D=3)
print('TPR: ', TPR,'\n',
      # 'FP: ', FP,'\n',
      # 'FN: ', FN,'\n',
      'MAE: ', MAE,'\n',
      'Precision: ', precision,'\n',
      'Recall: ', recall,'\n',
      'F1: ', F1)

# Room level scores
room_level_testing(dbscan_centroids_clean, gt_centroids_common, D=3)