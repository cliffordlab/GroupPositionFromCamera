#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:11:48 2023

@author: chegde
"""

import numpy as np
import os
import glob
import scipy.io


# Create .npy files
def create_npys(path):
    for p in os.listdir(path):
        if p[-3::] == 'mat':
            try:
                mat_data = scipy.io.loadmat(path+'/'+p)
                for k in mat_data.keys():
                    if k[0:2] == 'pi':
                        data = mat_data[k]
                        np_path = path + '/' + k + '.npy'
                        np.save(np_path, data)
                        #print(path+'/'+p)
            except:
                pass

# Check if .npy file exists for the day
root_path = '/labs/cliffordlab/data/EP6/openpose/2022/'
months = os.listdir(root_path)

for month in os.listdir(root_path):
    month_path = root_path + month
    for day in os.listdir(month_path):
        day_path = month_path + '/' + day
        print(month+'/'+day)
        for hour in os.listdir(day_path):
            hour_path = day_path + '/' + hour
            try: # To handle some hour directories that are not synced correctly
                for pi in os.listdir(hour_path):
                    pi_kpts_path = hour_path + '/' + pi 
                    if len(os.listdir(pi_kpts_path)) != 0:
                        pi_kpts_path += '/keypoints'
                    
                        # Check if pi_kpts_path has .npy files
                        npys = glob.glob(os.path.join(pi_kpts_path, '*.npy'))
                        
                        # If npys empty then create .npys file
                        if not npys:
                            #print('Creating')
                            create_npys(pi_kpts_path)
            except:
                pass
