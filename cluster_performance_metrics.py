#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:41:31 2023

@author: chegde
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import matplotlib.pyplot as plt

def cluster_metrics(detected_centroids, true_centroids, D=3):
    """
    Finds number of TP, FP and FN centroids detected.
    Centroid is considered correct detection if it is 
    detected within D meters of GT centroid. Detected 
    centroid is associated with the GT centroid it is
    closest to in terms of euclidean distance. It also
    gives the average deviation from GT centroids
    in terms of MAE.

    Parameters
    ----------
    detected_centroids : List of list of floats
        x,y positions of detected centroids.
    gt_centroids : List of list of floats
        x,y positions of ground truth 

    Returns
    -------
    Avg number of TP, FP, FN in each frame
    MAE across frames

    """
    
    # Convert D to pixel value
    D = D * 17.5
    
    tp_rate = []
    prec= []
    rec = []
    mean_abs_err = []
    f1_sc = []
    num_no_gt = 0
    for c_det, c_gt in zip(detected_centroids, true_centroids):
        
        if len(c_gt) == 0:
            num_no_gt += 1
            
        else:
        
            TP = 0
            FP = 0
            FN = 0
            MAE = 0
            
            num_det = len(c_det)
            num_gt = len(c_gt)
            
            # Create distance matrix (rows:gt, cols:det)
            distances = 1000*np.ones((num_gt, num_det))
            for j in range(num_gt):
                for k in range(num_det):
                    d = np.linalg.norm(np.array(c_gt[j])-np.array(c_det[k]))
                    if d <= D: # Update only if it is possible TP
                        distances[j,k] = d
            
            # Get detections closest to ground truths
            # Hungarian algorithm
            gt_match_temp, det_match_temp = linear_sum_assignment(distances)
            gt_match = []
            det_match = []
            distances_match = []
            for gt, det in zip(gt_match_temp, det_match_temp):
                if distances[gt,det] < 1000:
                    gt_match.append(gt)
                    det_match.append(det)
                    distances_match.append(distances[gt,det])
            gt_match = np.array(gt_match)
            det_match = np.array(det_match)
            distances_match = np.array(distances_match)
            
            TP += len(det_match)
            if num_det > len(det_match):
                FP += num_det - len(det_match)
            if num_gt > len(gt_match):
                FN += num_gt - len(gt_match)
            
            mae = 0
            count = 0
            for dd in distances_match:
                if not math.isnan(dd):
                    mae += dd
                    count += 1
            if count != 0:
                mae /= count
            MAE += mae
            
            if TP+FP == 0:
                P = 0
            else:
                P = TP/(TP+FP)
            if TP+FN == 0:
                R = 0
            else:
                R = TP/(TP+FN)
            if P==0 and R==0:
                f1 = 0
            else:
                f1 = 2*P*R/(P+R)
            
            TPR = TP/num_gt
            
            tp_rate.append(TPR)
            prec.append(P)
            rec.append(R)
            f1_sc.append(f1)
            if TP != 0:
                mean_abs_err.append(MAE)        
        
    TPR = np.mean(tp_rate)
    precision = np.mean(prec)
    recall = np.mean(rec)
    F1 = np.mean(f1_sc)
    MAE = np.mean(mean_abs_err)/17.5
    
    return TPR, precision, recall, F1, MAE

def room_level_testing(dbscan_centroids_clean, gt_centroids_common, D=3, return_type='print'):
    '''
    Reports performance metrics for each area in EP6.

    Parameters
    ----------
    dbscan_centroids_clean : List of list
        Estimated centroids. Each entry of list is centroids of one frame. Each inner list 
        is [2,] which contains (x,y) positions centroid.
    gt_centroids_common : Ground truth centroids
        Ground truth centroids corresponding to estimated centroids. Each entry 
        of list is centroids of one frame. Each inner list is [2,] which 
        contains (x,y) positions centroid.
    return_type : str, optional
        To print results use 'print', to return results as variables use 'return'.
        The default is 'print'.

    Returns
    -------
    precision : List
        List of precisions for each area in EP6.
    recall : List
        List of recalls for each area in EP6.
    f1 : List
        List of F1 scores for each area in EP6.

    '''
    # Coordinates to define rooms. Bottom left and top right coordinates defined.
    gym = [(78,670),(280,405)]
    act_std = [(78,405),(282,215)]
    dining = [(78,215),(440,90)]
    kitchen = [(440,215),(622,90)]
    lounge = [(410,673),(681,471)]
    techbar = [(683,670),(799,498)]
    grouproom = [(796,668),(1126,513)]
    staffzone = [(942,508),(1124,223)]
    
    det_cent_gym = []
    det_cent_act = []
    det_cent_dining = []
    det_cent_kitchen = []
    det_cent_lounge = []
    det_cent_techbar = []
    det_cent_grproom = []
    det_cent_staff = []
    for i in range(len(dbscan_centroids_clean)):
        det_cent_gym_frame = []
        det_cent_act_frame = []
        det_cent_dining_frame = []
        det_cent_kitchen_frame = []
        det_cent_lounge_frame = []
        det_cent_techbar_frame = []
        det_cent_grproom_frame = []
        det_cent_staff_frame = []
        for j in range(len(dbscan_centroids_clean[i])):
            if dbscan_centroids_clean[i][j][0] >= gym[0][0] and \
                dbscan_centroids_clean[i][j][0] <= gym[1][0] and \
                dbscan_centroids_clean[i][j][1] <= gym[0][1] and \
                dbscan_centroids_clean[i][j][1] >= gym[1][1]:
                    det_cent_gym_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= act_std[0][0] and \
                dbscan_centroids_clean[i][j][0] <= act_std[1][0] and \
                dbscan_centroids_clean[i][j][1] <= act_std[0][1] and \
                dbscan_centroids_clean[i][j][1] >= act_std[1][1]:
                    det_cent_act_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= dining[0][0] and \
                dbscan_centroids_clean[i][j][0] <= dining[1][0] and \
                dbscan_centroids_clean[i][j][1] <= dining[0][1] and \
                dbscan_centroids_clean[i][j][1] >= dining[1][1]: 
                    det_cent_dining_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= kitchen[0][0] and \
                dbscan_centroids_clean[i][j][0] <= kitchen[1][0] and \
                dbscan_centroids_clean[i][j][1] <= kitchen[0][1] and \
                dbscan_centroids_clean[i][j][1] >= kitchen[1][1]:
                    det_cent_kitchen_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= lounge[0][0] and \
                dbscan_centroids_clean[i][j][0] <= lounge[1][0] and \
                dbscan_centroids_clean[i][j][1] <= lounge[0][1] and \
                dbscan_centroids_clean[i][j][1] >= lounge[1][1]: 
                    det_cent_lounge_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= techbar[0][0] and \
                dbscan_centroids_clean[i][j][0] <= techbar[1][0] and \
                dbscan_centroids_clean[i][j][1] <= techbar[0][1] and \
                dbscan_centroids_clean[i][j][1] >= techbar[1][1]:    
                    det_cent_techbar_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= grouproom[0][0] and \
                dbscan_centroids_clean[i][j][0] <= grouproom[1][0] and \
                dbscan_centroids_clean[i][j][1] <= grouproom[0][1] and \
                dbscan_centroids_clean[i][j][1] >= grouproom[1][1]:    
                    det_cent_grproom_frame.append(dbscan_centroids_clean[i][j])
            if dbscan_centroids_clean[i][j][0] >= staffzone[0][0] and \
                dbscan_centroids_clean[i][j][0] <= staffzone[1][0] and \
                dbscan_centroids_clean[i][j][1] <= staffzone[0][1] and \
                dbscan_centroids_clean[i][j][1] >= staffzone[1][1]:
                    det_cent_staff_frame.append(dbscan_centroids_clean[i][j])
        if det_cent_gym_frame != []:
            det_cent_gym.append(det_cent_gym_frame)
        if det_cent_act_frame != []:
            det_cent_act.append(det_cent_act_frame)
        if det_cent_dining_frame != []:
            det_cent_dining.append(det_cent_dining_frame)
        if det_cent_kitchen_frame != []:
            det_cent_kitchen.append(det_cent_kitchen_frame)
        if det_cent_lounge_frame != []:
            det_cent_lounge.append(det_cent_lounge_frame)
        if det_cent_techbar_frame != []:
            det_cent_techbar.append(det_cent_techbar_frame)
        if det_cent_grproom_frame != []:
            det_cent_grproom.append(det_cent_grproom_frame)
        if det_cent_staff_frame != []:
            det_cent_staff.append(det_cent_staff_frame)
                    
    gt_cent_gym = []
    gt_cent_act = []
    gt_cent_dining = []
    gt_cent_kitchen = []
    gt_cent_lounge = []
    gt_cent_techbar = []
    gt_cent_grproom = []
    gt_cent_staff = []
    for i in range(len(gt_centroids_common)):
        gt_cent_gym_frame = []
        gt_cent_act_frame = []
        gt_cent_dining_frame = []
        gt_cent_kitchen_frame = []
        gt_cent_lounge_frame = []
        gt_cent_techbar_frame = []
        gt_cent_grproom_frame = []
        gt_cent_staff_frame = []
        for j in range(len(gt_centroids_common[i])):
            if gt_centroids_common[i][j][0] >= gym[0][0] and \
                gt_centroids_common[i][j][0] <= gym[1][0] and \
                gt_centroids_common[i][j][1] <= gym[0][1] and \
                gt_centroids_common[i][j][1] >= gym[1][1]:
                    gt_cent_gym_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= act_std[0][0] and \
                gt_centroids_common[i][j][0] <= act_std[1][0] and \
                gt_centroids_common[i][j][1] <= act_std[0][1] and \
                gt_centroids_common[i][j][1] >= act_std[1][1]:
                    gt_cent_act_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= dining[0][0] and \
                gt_centroids_common[i][j][0] <= dining[1][0] and \
                gt_centroids_common[i][j][1] <= dining[0][1] and \
                gt_centroids_common[i][j][1] >= dining[1][1]:
                    gt_cent_dining_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= kitchen[0][0] and \
                gt_centroids_common[i][j][0] <= kitchen[1][0] and \
                gt_centroids_common[i][j][1] <= kitchen[0][1] and \
                gt_centroids_common[i][j][1] >= kitchen[1][1]:
                    gt_cent_kitchen_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= lounge[0][0] and \
                gt_centroids_common[i][j][0] <= lounge[1][0] and \
                gt_centroids_common[i][j][1] <= lounge[0][1] and \
                gt_centroids_common[i][j][1] >= lounge[1][1]:
                    gt_cent_lounge_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= techbar[0][0] and \
                gt_centroids_common[i][j][0] <= techbar[1][0] and \
                gt_centroids_common[i][j][1] <= techbar[0][1] and \
                gt_centroids_common[i][j][1] >= techbar[1][1]:
                    gt_cent_techbar_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= grouproom[0][0] and \
                gt_centroids_common[i][j][0] <= grouproom[1][0] and \
                gt_centroids_common[i][j][1] <= grouproom[0][1] and \
                gt_centroids_common[i][j][1] >= grouproom[1][1]:
                    gt_cent_grproom_frame.append(gt_centroids_common[i][j])
            if gt_centroids_common[i][j][0] >= staffzone[0][0] and \
                gt_centroids_common[i][j][0] <= staffzone[1][0] and \
                gt_centroids_common[i][j][1] <= staffzone[0][1] and \
                gt_centroids_common[i][j][1] >= staffzone[1][1]:
                    gt_cent_staff_frame.append(gt_centroids_common[i][j])
        if gt_cent_gym_frame != []:
            gt_cent_gym.append(gt_cent_gym_frame)
        if gt_cent_act_frame != []:
            gt_cent_act.append(gt_cent_act_frame)
        if gt_cent_dining_frame != []:
            gt_cent_dining.append(gt_cent_dining_frame)
        if gt_cent_kitchen_frame != []:
            gt_cent_kitchen.append(gt_cent_kitchen_frame)
        if gt_cent_lounge_frame != []:
            gt_cent_lounge.append(gt_cent_lounge_frame)
        if gt_cent_techbar_frame != []:
            gt_cent_techbar.append(gt_cent_techbar_frame)
        if gt_cent_grproom_frame != []:
            gt_cent_grproom.append(gt_cent_grproom_frame)
        if gt_cent_staff_frame != []:
            gt_cent_staff.append(gt_cent_staff_frame)
       
    
    TPR_gym, precision_gym, recall_gym, F1_gym, MAE_gym = cluster_metrics(detected_centroids=det_cent_gym, 
                                                      true_centroids=gt_cent_gym, D=D)
    
    TPR_act, precision_act, recall_act, F1_act, MAE_act = cluster_metrics(detected_centroids=det_cent_act, 
                                                      true_centroids=gt_cent_act, D=D)
    
    TPR_dining, precision_dining, recall_dining, F1_dining, MAE_dining = cluster_metrics(detected_centroids=det_cent_dining, 
                                                      true_centroids=gt_cent_dining, D=D)
    
    TPR_kitchen, precision_kitchen, recall_kitchen, F1_kitchen, MAE_kitchen = cluster_metrics(detected_centroids=det_cent_kitchen, 
                                                      true_centroids=gt_cent_kitchen, D=D)
    
    TPR_lounge, precision_lounge, recall_lounge, F1_lounge, MAE_lounge = cluster_metrics(detected_centroids=det_cent_lounge, 
                                                      true_centroids=gt_cent_lounge, D=D)
    
    TPR_techbar, precision_techbar, recall_techbar, F1_techbar, MAE_techbar = cluster_metrics(detected_centroids=det_cent_techbar, 
                                                      true_centroids=gt_cent_techbar, D=D)
    
    TPR_grproom, precision_grproom, recall_grproom, F1_grproom, MAE_grproom = cluster_metrics(detected_centroids=det_cent_grproom, 
                                                      true_centroids=gt_cent_grproom, D=D)
    
    TPR_staff, precision_staff, recall_staff, F1_staff, MAE_staff= cluster_metrics(detected_centroids=det_cent_staff, 
                                                      true_centroids=gt_cent_staff, D=D)
    
    if return_type == 'print':
        print('TPR gym: ', TPR_gym,'\n',
              'MAE gym: ', MAE_gym,'\n',
              'Precision gym: ', precision_gym,'\n',
              'Recall gym: ', recall_gym,'\n',
              'F1 gym: ', F1_gym)
        print('TPR activity area: ', TPR_act,'\n',
              'MAE activity area: ', MAE_act,'\n',
              'Precision activity area: ', precision_act,'\n',
              'Recall activity area: ', recall_act,'\n',
              'F1 activity area: ', F1_act)
        print('TPR dining: ', TPR_dining,'\n',
              'MAE dining: ', MAE_dining,'\n',
              'Precision dining: ', precision_dining,'\n',
              'Recall dining: ', recall_dining,'\n',
              'F1 dining: ', F1_dining)
        print('TPR kitchen: ', TPR_kitchen,'\n',
              'MAE kitchen: ', MAE_kitchen,'\n',
              'Precision kitchen: ', precision_kitchen,'\n',
              'Recall kitchen: ', recall_kitchen,'\n',
              'F1 kitchen: ', F1_kitchen)
        print('TPR lounge: ', TPR_lounge,'\n',
              'MAE lounge: ', MAE_lounge,'\n',
              'Precision lounge: ', precision_lounge,'\n',
              'Recall lounge: ', recall_lounge,'\n',
              'F1 lounge: ', F1_lounge)
        print('TPR techbar: ', TPR_techbar,'\n',
              'MAE techbar: ', MAE_techbar,'\n',
              'Precision techbar: ', precision_techbar,'\n',
              'Recall techbar: ', recall_techbar,'\n',
              'F1 techbar: ', F1_techbar)
        print('TPR grproom: ', TPR_grproom,'\n',
              'MAE grproom: ', MAE_grproom,'\n',
              'Precision grproom: ', precision_grproom,'\n',
              'Recall grproom: ', recall_grproom,'\n',
              'F1 grproom: ', F1_grproom)
        print('TPR staff: ', TPR_staff,'\n',
              'MAE staff: ', MAE_staff,'\n',
              'Precision staff: ', precision_staff,'\n',
              'Recall staff: ', recall_staff,'\n',
              'F1 staff: ', F1_staff)
    
    elif return_type == 'return':
        precision = [precision_gym, precision_act, precision_dining, precision_kitchen, precision_lounge, precision_techbar, precision_grproom, precision_staff]
        recall = [recall_gym, recall_act, recall_dining, recall_kitchen, recall_lounge, recall_techbar, recall_grproom, recall_staff]
        f1 = [F1_gym, F1_act, F1_dining, F1_kitchen, F1_lounge, F1_techbar, F1_grproom, F1_staff]
        mae = [MAE_gym, MAE_act, MAE_dining, MAE_kitchen, MAE_lounge, MAE_techbar, MAE_grproom, MAE_staff]/17.5
        
        return precision, recall, f1, mae
    

 
