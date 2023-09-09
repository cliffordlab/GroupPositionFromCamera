#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:03:30 2022

@author: chegde
"""

import numpy as np
from collections import defaultdict
import math

# Create ground truth labels
def April2022_groundtruth():
    
    ground_truth = []
    # Handcode framewise
    gt_dict = {0:0, 1:0, 2:0, 3:1, 4:1} # Frame 1
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:0, 2:0, 3:1, 4:1} # Frame 2
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:0, 2:0, 3:1, 4:2} # Frame 3
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:1, 2:1, 3:1, 4:2} # Frame 4
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:0, 2:0, 3:1, 4:2} # Frame 5
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:0, 2:0, 3:1, 4:2} # Frame 6
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:0, 2:1, 3:1, 4:1} # Frame 7
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:1, 2:1, 3:2, 4:3} # Frame 8
    ground_truth.append(gt_dict)
    gt_dict = {0:0, 1:1, 2:2, 3:1, 4:2} # Frame 9
    ground_truth.append(gt_dict)
    
    return ground_truth


def performance_metrics(ground_truth, detections):
    
    precision = []
    recall = []
    for i in range(len(ground_truth)):
        GT, gt_keys_sorted = change_grouping_style(ground_truth[i])
        DET, det_keys_sorted = change_grouping_style(detections[i])
        
        num_grps = len(GT)
        num_detected_grps = len(DET)
        num_tp_grps_in_frame = 0
        for j in range(num_grps):
            GT_grp = GT[gt_keys_sorted[j]]
            cardinality = len(GT_grp)
            tp = math.ceil(2*cardinality/3) - 2
            fp = math.floor(cardinality/3) + 1
            for k in range(num_detected_grps):
                DET_grp = DET[det_keys_sorted[k]]
                # Check how many members of this detected group are actually correct
                correct_count = len(list(set(GT_grp) & set(DET_grp)))
                incorrect_count = max(0, len(DET_grp)-len(GT_grp))
                if correct_count >= tp and incorrect_count <= fp:
                    tp = correct_count
                    fp = incorrect_count
            # Based on tp & fp, determine if group is correctly detected
            if tp >= math.ceil(2*cardinality/3) and fp <= math.floor(cardinality/3):
                num_tp_grps_in_frame += 1
                
        P = num_tp_grps_in_frame/num_detected_grps
        R = num_tp_grps_in_frame/num_grps
        precision.append(P)
        recall.append(R)
    
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = (2*avg_precision*avg_recall)/(avg_precision+avg_recall)
    
    return avg_precision, avg_recall, avg_f1

def find_closest(val1, val2, target):
    return val2 if target - val1 >= val2 - target else val1

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
                x = (x1-77)/17.5
                y = (669-y1)/17.5
                # x1 = get_closest_value(xs, x1)
                # y1 = get_closest_value(ys, y1)
                # a = list(xs).index(x1)
                # b = list(ys).index(y1)
                
                # x = int(a - 1)
                # y = int(34 - b - 1)
                
                # Handle y axis being flipped because (0,0) is now
                # the top left corner instead of bottom left (how
                # it was for ground truth annotations)
                #y = -(y - 33)
                
                pos = [x,y]
                frame_pos.append(pos)
            rescaled_positions.append(frame_pos)
    
    return rescaled_positions

def change_grouping_style(groups):
    
    v = defaultdict(list)
    
    for key, value in sorted(groups.items()): # Key is grp number. Value is people in grp
        v[value].append(key)
    
    sorted_keys = []
    for k in sorted(v, key=lambda k: len(v[k]), reverse=True):
        sorted_keys.append(k)
    
    return v, sorted_keys