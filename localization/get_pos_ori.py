#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:45:28 2023

@author: chegde
"""

import numpy as np
#from utils.tracking import *
from localization.funcs import load_data, kps_fp_1, kps_smth_kf, occup_ori_est, occup_ori_smth_kf, occup_ori_mv, mv_smth_kf
from datetime import datetime, timedelta
import config as cfg
import matplotlib.pyplot as plt
import math


#%%%%%%%%%%%%% Process data %%%%%%%%%%%%%%%%%%#
def get_pos_ori(start_time, end_time):
    # Load projection matrices for each camera
    dir_proj_mat = cfg.dir_exp + '/proj_mat'
    cfg.dir_proj_mat = dir_proj_mat
    
    # # Load MEBOW model
    # model, dataloader = load_mebow_model(cfg)
    
    # Load data
    pi_data = load_data(start_time, end_time, cfg)
    
    # Preprocess data
    pi_data = kps_fp_1(pi_data, cfg) # Remove poses that don't change in consecutive frames
    pi_data, pi_pid_seq, kps_seq_pi_pid_np = kps_smth_kf(pi_data, cfg) # Kalman filter based filling
    
    # Get positions and orientations
    occup_pi, ori_pi_mebow, ori_pi_2d = occup_ori_est(pi_data, pi_pid_seq, kps_seq_pi_pid_np, cfg)
    
    # Smoothen positions and orientations using kalman filter
    occup_pi, ori_pi_mebow = occup_ori_smth_kf(occup_pi, ori_pi_mebow, ori_pi_2d, pi_data, cfg)
    
    # Multi-view person association across all cameras
    occup_mv, ori_mv = occup_ori_mv(occup_pi, ori_pi_mebow, pi_data, cfg)
    
    # Tracking
    mv_seq = mv_smth_kf(occup_mv, ori_mv, cfg)
    
    
    #%%%%%%%%%% Convert tracking position and orientations to required format %%%%%%%%%%%#
    time_stepsize = 1 # 1 second
    delta = timedelta(seconds=time_stepsize)
    times = []
    while start_time < end_time:
        times.append(start_time)
        start_time += delta
        
    def vec_to_angle(o):
        rad_pred = np.arctan2(o[1], o[0])
        # print (np.degrees(rad_pred))
        degree_pred = np.degrees(rad_pred)%360 #+ 90
        # print (degree_pred)
        # if degree_pred < 0:
        #   degree_pred += 360
        #degree_pred = 360 - degree_pred
        return degree_pred
    
    positions = []
    orientations = []
    timestamps = []
    keypoints = []
    for t in times:
        loc = []
        ori = []
        for pid in mv_seq:
            tm = mv_seq[pid]['dt']
            if t in tm:
              i = tm.index(t)  
              l = mv_seq[pid]['loc'][i][0]
              o = mv_seq[pid]['ori'][i][0]
    
              theta = vec_to_angle(o)
              
              loc.append(l)
              ori.append(theta)
    
        positions.append(loc)
        orientations.append(ori)
        timestamps.append(t)
        
    return positions, orientations, timestamps
    
