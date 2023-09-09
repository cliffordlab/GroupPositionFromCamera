#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:19:50 2022

@author: chegde
"""

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import ast
    

def create_timesequences(start_time, stop_time):
    start_time = datetime.strptime(start_time, '%Y%m%d_%H%M%S')
    stop_time = datetime.strptime(stop_time, '%Y%m%d_%H%M%S')
    time_stepsize = 1 # 1 second
    delta = timedelta(seconds=time_stepsize)
    times = []
    while start_time < stop_time:
        times.append(start_time)
        start_time += delta
    return times

def ground_truth_groups(gt_path='sample_data/group_ground_truth.xlsx'):
        
    data = pd.read_excel(gt_path)
    data = data.to_numpy()
    
    GT = []
    start_flag = 0
    for d in data:
        s = '20221118_' + str(int(d[0]))
        e = '20221118_' + str(int(d[1]))
        time_range = create_timesequences(s, e)
        grps = {0:d[2], 1:d[3], 2:d[4], 3:d[5], 4:d[6], 5:d[7], 6:d[8]}
        grps_repeat = [grps] * len(time_range)
        
        if start_flag == 0:
            GT = [time_range, grps_repeat]
            start_flag = 1
        else:
            GT = np.hstack((GT, [time_range, grps_repeat]))
    
    return GT

def ground_truth_centroids(gt_path='sample_data/group_centroids_ground_truth.xlsx'):
    
    data =pd.read_excel(gt_path)
    data = data.to_numpy()
    
    gt_centroids = []
    start_flag = 0
    for d in data:
        s = '20221118_' + str(int(d[0]))
        e = '20221118_' + str(int(d[1]))
        time_range = create_timesequences(s, e)
        cent = ast.literal_eval(d[3]) # Change to 2 once new labeling done
        # Convert to pixel coordinates
        cent_pixels = []
        for c in cent:
            x, y = c
            x = 17.5*x + 77
            y = 669 - 17.5*y
            cent_pixels.append([x,y])
            
        cent_repeat = [cent_pixels] * len(time_range)
        
        if start_flag == 0:
            gt_centroids = [time_range, cent_repeat]
            start_flag = 1
        else:
            gt_centroids[0] = gt_centroids[0] + time_range
            gt_centroids[1] = gt_centroids[1] + cent_repeat
            
    return gt_centroids