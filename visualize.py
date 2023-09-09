#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 20:15:58 2023

@author: chegde
"""

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt5
import math
import time

floorplan_path = '/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/group_detection/ep6_floorplan_measured_half_gridded_1_meter.jpg'
floorplan = plt.imread(floorplan_path)

xs = np.arange(77, 1129, 17.5)
ys = np.arange(90, 669, 17.5)

def pos_ori(pos, ori):
    
    """
    Display positions and orientations of individuals on the EP6 map.
    
    Parameters:
        pos: list of array of positions of individuals in a single timestep. The 
            array contains the (x,y) coordinates of the position. The coordinates
            are in terms of pixels and not meters.
        ori: list of orientations of individuals in a single timestep in degrees.
    """
    
    line_len = 15
    
    plt.figure()
    plt.imshow(floorplan)
    
    for p, o in zip(pos,ori): # p & o are position and orientation of one person in a frame
        x, y = p
        # x = xs[int(x)+1]
        # y = ys[34-int(y)-1]
        
        # o = -1*o + 90
        # # print (degree_pred)
        # if o < 0:
        #   o += 360
        
        endy = y - line_len * math.sin(math.radians(o))
        endx = x + line_len * math.cos(math.radians(o))
            
        plt.plot([x, endx], [y, endy], color='b', linewidth=2)
        plt.plot(x, y, 'o', color='b', markersize=4)
 