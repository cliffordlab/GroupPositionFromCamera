#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:54:43 2023

@author: chegde
"""

import pickle as cp
import numpy as np


def April2022_GT(ordered_by_frames=True):

        # Load GT location and orientation file
        gt_path = '/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/localization/annot_collection.p'
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
        
        return positions, orientations
    
def position_rescale(positions):
    
    xs = np.arange(77, 1129, 17.5)
    ys = np.arange(90, 669, 17.5)
    
    rescaled_positions = []
    for p in positions:
        if not p:
            rescaled_positions.append(p)
        else:
            frame_pos = []
            for pp in p:
                x1, y1 = pp
                x1 = get_closest_value(xs, x1)
                y1 = get_closest_value(ys, y1)
                a = list(xs).index(x1)
                b = list(ys).index(y1)
                
                x = int(a - 1)
                y = int(34 - b - 1)
                
                # Handle y axis being flipped because (0,0) is now
                # the top left corner instead of bottom left (how
                # it was for ground truth annotations)
                #y = -(y - 33)
                
                pos = [x,y]
                frame_pos.append(pos)
            rescaled_positions.append(frame_pos)
    
    return rescaled_positions

def get_closest_value(arr, target):
    n = len(arr)
    left = 0
    right = n - 1
    mid = 0

    # edge case - last or above all
    if target >= arr[n - 1]:
        return arr[n - 1]
    # edge case - first or below all
    if target <= arr[0]:
        return arr[0]
    # BSearch solution: Time & Space: Log(N)

    while left < right:
        mid = (left + right) // 2  # find the mid
        if target < arr[mid]:
            right = mid
        elif target > arr[mid]:
            left = mid + 1
        else:
            return arr[mid]

    if target < arr[mid]:
        return find_closest(arr[mid - 1], arr[mid], target)
    else:
        return find_closest(arr[mid], arr[mid + 1], target)
    
def find_closest(val1, val2, target):
    return val2 if target - val1 >= val2 - target else val1