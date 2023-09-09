#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 00:45:56 2023

@author: chegde
"""

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load floorplan image
floor_plan = plt.imread('ep6_floorplan_measured_half_gridded_1_meter.jpg')

def create_frustum(p, o):
    """
    Create weighted frustum for an individual given their position and orientation.

    Parameters
    ----------
    p : Array or list of size (2,)
        (x,y) position of individual.
    o : float
        Orientation of individual.

    Returns
    -------
    frustum_collector : 2D array
        2D array of shape equal to x,y dimensions of floorplan which has all elements
        set to zero except the area where frustum is created. Frustum is layed
        out on this map.

    """
    
    y, x = p # Positions
    l=2.1 # 2.1m is the edge of close social interaction distance. We don't 
            # consider people beyond this distance to be interacting.
    gamma=160 # Range of eye motion or visibility
    sigma = gamma/6 #(1/3)*(gamma/2) spread of Gaussian curve so that interaction is 
                    # considered to be within eye range.
                    
    # Initialize frustum collector
    frustum_collector = np.zeros((np.shape(floor_plan)[0], np.shape(floor_plan)[1]))
    
    start_angle = o - int(gamma/2)
    end_angle = o + int(gamma/2)  
    angles = np.linspace(start_angle, end_angle, 80) # Define angles
    
    L = l*17.5 # 17.5 pixels = 1m
    distances = np.linspace(0.01,L,80) # Define distances from person
    
    #### Get weights - Gaussian for angles, normalized Euclidean for distance ####
    Wa = stats.norm.pdf(angles, loc=o, scale=sigma)
    angles = (-1*angles + 270)%360
    
    Wd = 1/distances
    W = Wa[:,np.newaxis]*Wd
    
    cos_angles = np.cos(np.radians(angles))
    sin_angles = -np.sin(np.radians(angles))
    sx = np.outer(cos_angles, distances) + x
    sy = np.outer(sin_angles, distances) + y
    # # Round and convert to integer indices
    sx_int = np.int_(sx) 
    sy_int = np.int_(sy) 
    # Clip indices to stay within bounds
    sx_int = np.clip(sx_int, 0, frustum_collector.shape[0] - 1)
    sy_int = np.clip(sy_int, 0, frustum_collector.shape[1] - 1)
    frustum_collector[sx_int,sy_int] = W
            
    return frustum_collector

def eps_frustum(max_dist):
    """
    Calculate eps value when using pos+ori for group localization.

    Parameters
    ----------
    max_dist : Float
        Maximum Euclidean distance between two people that is considered interaction.

    Returns
    -------
    dist_max : Float
        Maximum pos+ori distance which is eps value for DBSCAN.

    """
    
    frustum1 = create_frustum([200,200],30)
    frustum2 = create_frustum([200,200+(max_dist)],0) 
    dist_max = weight_calculation(frustum1, frustum2)
    
    return dist_max
    
def weight_calculation(frustum_i, frustum_j):
    """
    Calculate pos+ori distance between two individuals, which is the overlapping 
    weight between frustums of two individuals.

    Parameters
    ----------
    frustum_i : 2D array
        Frustum of person i.
    frustum_j : 2D array
        Frustum of person j.

    Returns
    -------
    distance : Float
        Pos+ori distance between two individuals.

    """
    
    # Find amount of overlap of frustums 
    non_zero_mask = np.logical_and(frustum_i!=0, frustum_j!=0).astype(int)
    joint_frustum = frustum_i + frustum_j
    joint_frustum = joint_frustum * non_zero_mask
    joint_frustum_value = np.sum(joint_frustum)
    if joint_frustum_value != 0:
        distance = 1/joint_frustum_value
    else:
        distance = 1000
    
    return distance

def pos_ori_frustum_dist_mat(pos, ori):
    """
    Create distance matrix using position and orientations

    Parameters
    ----------
    pos : List of arrays. Each array is of size (2,). 
        Positions of all people in one frame. Each array of size (2,) contains
        the (x,y) positions of a person.  
    ori : List
        Orientations of all people in one frame corresponding to positions.

    Returns
    -------
    distance_matrix : 2D array of size (len(pos),len(pos))
        Matrix containing pos+ori distance between each individual

    """
    
    distance_matrix = 1000*np.ones((len(pos),len(pos)))
    
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j:
                frustum_i = create_frustum(pos[i], ori[i])
                frustum_j = create_frustum(pos[j], ori[j])
                distance = weight_calculation(frustum_i, frustum_j)
                distance_matrix[i,j] = distance
                
    return distance_matrix



def dbscan_clusters(frame_positions, frame_orientations, eps, only_positions=True):
    """
    Perform DBSCAN clustering to identify and localize groups.

    Parameters
    ----------
    frame_positions : List of arrays. Each array is of size (2,). 
        Positions of all people in one frame. Each array of size (2,) contains
        the (x,y) positions of a person.
    frame_orientations : List
        Orientations of all people in one frame corresponding to positions.
    eps : Float
        Eps parameter for DBSCAN. Indicates the maximum distance that is
        considered as interaction.
    only_positions : Bool, optional
        If True, use only positions for DBSCAN. If False use positions and 
        orientations for DBSCAN. The default is True.

    Returns
    -------
    groups : Dictionary
        Keys are person ID, values are their estimated number of group membership.
    centroids : List of lists
        List of position of all centroids in the frame. The inner list has two
        elements which are the x and y positions of the centroid.

    """
    
    if only_positions:
        clustering = DBSCAN(eps=eps, min_samples=2).fit(frame_positions)
        clusters_labels = clustering.labels_
            
    else:
        distance_matrix = pos_ori_frustum_dist_mat(frame_positions, frame_orientations) # Create distance matrix
        eps = eps_frustum(eps) # Get eps value for pos+ori
        clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(distance_matrix)
        clusters_labels = clustering.labels_  
        
    # Convert grouping into dictionary
    groups = {}
    for i in range(len(clusters_labels)):
        groups[i] = clusters_labels[i]
        
    # Get centroids of groups
    centroids = []
    unique_grps = list(np.unique(clusters_labels))
    for i in range(len(unique_grps)):
        cluster_pplX = []
        cluster_pplY = []
        for j in range(len(clusters_labels)):
            if clusters_labels[j] == unique_grps[i]:
                cluster_pplX.append(frame_positions[j][0])
                cluster_pplY.append(frame_positions[j][1])
        X = np.mean(cluster_pplX)
        Y = np.mean(cluster_pplY)
        centroids.append([X,Y])
    
    return groups, centroids