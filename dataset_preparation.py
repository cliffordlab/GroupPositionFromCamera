#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:19:50 2022

@author: chegde
"""

import numpy as np
import csv
import os
import math
from util import April2022_groundtruth
import pickle as cp
import pandas as pd
from datetime import timedelta, datetime
import ast

class PrepareDatasets():
    
    def __init__(self):
        pass
    
    # def salsa(self):
    #     ######### Load group annotations ########
    #     label_path = '/home/chegde/Cliffordlab/CEP/F-formations/group_detection/public_datasets/salsa/SALSA_Annotation_ps/Annotation/salsa_ps/fformationGT.csv'
        
    #     annotations = []
    #     with open(label_path, 'r') as file:
    #         reader = csv.reader(file)
    #         for lines in reader:
    #             annotations.append(lines)
                
    #     # Get list of time stamps
    #     all_timestamps = []
    #     for i in range(len(annotations)):
    #         all_timestamps.append(float(annotations[i][0]))
    #     timestamps = np.unique(all_timestamps)
        
    #     # Prepare GT group labels 
    #     ground_truth = []
    #     i = 0
    #     for t in range(len(timestamps)):
    #         groups = {}
    #         grp_counter = 0
    #         while i<len(annotations) and float(annotations[i][0]) == timestamps[t]:
    #             for j in range(1,len(annotations[i])):
    #                 groups[int(annotations[i][j])-1] = grp_counter
    #             grp_counter += 1
    #             i += 1
    #         ground_truth.append(groups)
                     
            
    #     ######### Load positions and orientations #########
    #     pos_ori_path = '/home/chegde/Cliffordlab/CEP/F-formations/group_detection/public_datasets/salsa/SALSA_Annotation_ps/Annotation/salsa_ps/geometryGT'
        
    #     person_positions = []
    #     person_orientations = []
    #     for file in os.listdir(pos_ori_path):
    #         # Load file
    #         pos_ori = []
    #         path = pos_ori_path + '/' + file
    #         with open(path, 'r') as f:
    #             reader = csv.reader(f)
    #             for lines in reader:
    #                 pos_ori.append(lines)
                    
    #         pos = []
    #         ori = []
    #         for i in range(len(pos_ori)):
    #             X = float(pos_ori[i][1])
    #             Y = float(pos_ori[i][2])
    #             angle = math.degrees(float(pos_ori[i][4]))
    #             pos.append([X,Y])
    #             ori.append(angle)
                
    #         person_positions.append(pos)
    #         person_orientations.append(ori)
            
    #     # Convert positions, orientations to ordering required
    #     positions = []
    #     orientations = []
    #     num_ppl = len(person_positions)
    #     for i in range(num_ppl):
    #         pos = []
    #         ori = []
    #         for t in range(len(timestamps)):
    #             pos.append(person_positions[i][t])
    #             ori.append(person_orientations[i][t])
    #         positions.append(pos)
    #         orientations.append(ori)
            
    #     return positions, orientations, ground_truth
    
    def April2022(self, ordered_by_frames=True):

        # Load GT location and orientation file
        gt_path = '/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/annot_collection.p'
        gt = cp.load(open(gt_path, 'rb'))

        gt_orientation = gt['Orientation']
        gt_orientation = gt_orientation['orientation']

        positions = []
        orientations = []
        group_num = []
        for k in gt_orientation.keys():
            if 'person' in k:
                if 'group' in k:
                    group_num.append(gt_orientation[k])
                elif 'orientation' in k:
                    orientations.append(gt_orientation[k])
                else:
                    positions.append(gt_orientation[k])
                    
        # Order by frames
        if ordered_by_frames==True:
            pos_frames = []
            ori_frames = []
            for t in range(len(positions[0])):
                p = []
                o = []
                for i in range(len(positions)):
                    p.append(positions[i][t])
                    o.append(orientations[i][t])
                pos_frames.append(p)
                ori_frames.append(o)
            positions = pos_frames
            orientations = ori_frames
        
        # Get ground truth annotations
        ground_truth = April2022_groundtruth()
        
        return positions, orientations, ground_truth
    
    def angle_math_to_graphical(self, theta):
        theta = 90 - theta
        if theta < 0:
            theta = 360 + theta
        return theta
    
    def April2022_clusters(self, ordered_by_frames=True):
        
        # Load GT location and orientation file
        gt_path = '/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/annot_collection.p'
        gt = cp.load(open(gt_path, 'rb'))

        gt_orientation = gt['Orientation']
        gt_orientation = gt_orientation['orientation']

        positions = []
        orientations = []
        group_num = []
        for k in gt_orientation.keys():
            if 'person' in k:
                if 'group' in k:
                    group_num.append(gt_orientation[k])
                elif 'orientation' in k:
                    orientations.append(gt_orientation[k])
                else:
                    positions.append(gt_orientation[k])
                    
        # Order by frames
        if ordered_by_frames==True:
            pos_frames = []
            ori_frames = []
            for t in range(len(positions[0])):
                p = []
                o = []
                for i in range(len(positions)):
                    x,y = positions[i][t]
                    x = 17.5*x - 77
                    y = 669 - 17.5*y
                    p.append([x,y])
                    o.append(self.angle_math_to_graphical(orientations[i][t]))
                pos_frames.append(p)
                ori_frames.append(o)
            positions = pos_frames
            orientations = ori_frames
        
        # Hand annotated centroid positions
        centroids = [[[8,11.5],[8.5,18]],
                     [[8,11.5],[8.5,18]],
                     [[8,11.5],[8.5,18]],
                     [[9,5],[5,4]],
                     [[9,5],[5,4]],
                     [[9,5],[5,4]],
                     [[10,24.5], [7.5,24.5]],
                     [[9,24.5],[7,25],[11,25]],
                     [[7,25],[9,24],[10,25]]]
        
        # Rescale centroids to pixel values
        centroids_rescaled = []
        for c in centroids:
            temp_cent = []
            for i in range(len(c)):
                x,y = c[i]
                x = 17.5*x - 77
                y = 669 - 17.5*y
                temp_cent.append([x,y])
            centroids_rescaled.append(temp_cent)
                
        return positions, orientations, centroids_rescaled
            
    
    def create_timesequences(self, start_time, stop_time):
        start_time = datetime.strptime(start_time, '%Y%m%d_%H%M%S')
        stop_time = datetime.strptime(stop_time, '%Y%m%d_%H%M%S')
        time_stepsize = 1 # 1 second
        delta = timedelta(seconds=time_stepsize)
        times = []
        while start_time < stop_time:
            times.append(start_time)
            start_time += delta
        return times
    
    def Nov2022(self, gt_path='/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/group_labels.xlsx'):
        
        data = pd.read_excel(gt_path)
        data = data.to_numpy()
        
        GT = []
        start_flag = 0
        for d in data:
            s = '20221118_' + str(int(d[0]))
            e = '20221118_' + str(int(d[1]))
            time_range = self.create_timesequences(s, e)
            grps = {0:d[2], 1:d[3], 2:d[4], 3:d[5], 4:d[6], 5:d[7], 6:d[8]}
            grps_repeat = [grps] * len(time_range)
            
            if start_flag == 0:
                GT = [time_range, grps_repeat]
                start_flag = 1
            else:
                GT = np.hstack((GT, [time_range, grps_repeat]))
        
        return GT
    
    def Nov2022_clusters(self, gt_path='/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/group_centroids_new.xlsx'):
        
        data =pd.read_excel(gt_path)
        data = data.to_numpy()
        
        gt_centroids = []
        start_flag = 0
        for d in data:
            s = '20221118_' + str(int(d[0]))
            e = '20221118_' + str(int(d[1]))
            time_range = self.create_timesequences(s, e)
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
    
    def Built_env_clusters(self, gt_path='/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/built_env_dataset.xlsx'):
        
        data = pd.read_excel(gt_path)
        data = data.to_numpy()
        
        dates = []
        time_intervals = []
        times = []
        datetimes = []
        positions = []
        gt_positions = []
        for i in range(len(data)):
            dt = data[i][0][0:8]
            month = dt[0:2]
            day = dt[3:5]
            year = '20'+dt[6:8]
            date = datetime.strptime(year+month+day, '%Y%m%d')
            dates.append(date)
            
            time_interval = data[i][5]
            start_time, end_time = time_interval.split('-')
            start_time = start_time + ':00'
            end_time = end_time + ':00'
            start_time = start_time.replace(" ", "")
            end_time = end_time.replace(" ", "")
            
            start_time = datetime.strptime(start_time, '%H:%M:%S')
            start_time = start_time + timedelta(seconds=60)
            start_time = start_time.strftime('%H:%M:%S')
            
            end_time = datetime.strptime(end_time, '%H:%M:%S')
            end_time = end_time + timedelta(seconds=60)
            end_time = end_time.strftime('%H:%M:%S')
            # start_tm = datetime.strptime(start_time, '%H:%M').time()
            # end_tm = datetime.strptime(end_time, '%H:%M').time()
            # start_hour = start_tm.hour
            # start_min = start_tm.minute
            # end_hour = end_tm.hour
            # end_min = end_tm.minute
            start_hour, start_min, start_sec = start_time.split(':')
            end_hour, end_min, end_sec = end_time.split(':')
            
            # start_datetime = datetime.combine(date.date(), start_tm)
            # start_time = start_datetime.replace(second=0) + timedelta(seconds=60)
            
            # end_datetime = datetime.combine(date.date(), end_tm)
            # end_time = end_datetime.replace(second=0) + timedelta(seconds=-60)
            
            start_datetime = year+month+day+'_'+start_hour+start_min+start_sec
            end_datetime = year+month+day+'_'+end_hour+end_min+end_sec
            
            time_range = self.create_timesequences(start_datetime, end_datetime)
            
            pos_frame = []
            for j in range(7,39):
                if isinstance(data[i][j], str):
                    pos = data[i][j]
                    pos = tuple(map(int, pos.split(',')))
                    pos_frame.append(pos)
            positions.append(pos_frame)
            
            positions_repeat = [pos_frame] * len(time_range)
            
            if i == 0:
                gt_positions = [time_range, positions_repeat]
            else:
                gt_positions[0] = gt_positions[0] + time_range
                gt_positions[1] = gt_positions[1] + positions_repeat
                
        return gt_positions
    
    def Built_env(self, gt_path='/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/built_env_dataset.xlsx'):
        
        data = pd.read_excel(gt_path)
        data = data.to_numpy()
        
        gt_positions = []
        gt_timestamps = []
        for i in range(len(data)):
            dt = data[i][0][0:8]
            month = dt[0:2]
            day = dt[3:5]
            year = '20'+dt[6:8]
            
            time_interval = data[i][5]
            start_time, end_time = time_interval.split('-')
            start_time = start_time + ':00'
            end_time = end_time + ':00'
            start_time = start_time.replace(" ", "")
            end_time = end_time.replace(" ", "")
            
            start_time = datetime.strptime(start_time, '%H:%M:%S')
            start_time = start_time + timedelta(seconds=60)
            start_time = start_time.strftime('%H:%M:%S')
            
            end_time = datetime.strptime(end_time, '%H:%M:%S')
            end_time = end_time + timedelta(seconds=-60)
            end_time = end_time.strftime('%H:%M:%S')
            
            start_hour, start_min, start_sec = start_time.split(':')
            end_hour, end_min, end_sec = end_time.split(':')
            
            start_datetime = year+month+day+'_'+start_hour+start_min+start_sec
            end_datetime = year+month+day+'_'+end_hour+end_min+end_sec
            
            time_range = self.create_timesequences(start_datetime, end_datetime)
            
            pos_frame = []
            for j in range(7,39):
                if isinstance(data[i][j], str):
                    pos = data[i][j]
                    pos = tuple(map(int, pos.split(',')))
                    pos_frame.append(pos)
            
            positions_repeat = [pos_frame] * len(time_range)
            
            gt_positions.append(positions_repeat)
            gt_timestamps.append(time_range)
                
        return gt_positions, gt_timestamps