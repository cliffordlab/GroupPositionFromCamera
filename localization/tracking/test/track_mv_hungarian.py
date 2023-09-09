'''
Multi-view tracking code
'''

import sys
import enum
import os
import pickle as cp
from tracemalloc import start
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import cv2
import scipy.io
from pprint import pprint
from scipy.optimize import linear_sum_assignment
import pandas as pd
from pandas import Series
import shutil
import copy
from itertools import groupby
from operator import itemgetter
from datetime import datetime, timedelta
import networkx as nx
import gc
import torch

from utils.tracking import *

frame_height = 481
frame_width = 641

# dir_root = '/home/hhyeokk/Research/CEP' # oddjob
# dir_data = '/labs/cliffordlab/data/EP6/openpose' # oddjobs
# dir_exp = dir_root + '/exps'
# dir_proj_mat = '/Users/jigardesai/Downloads/proj_mat' # pushti local

# dir_exp = '/home/hhyeokk/Research/logs/cams/track_mv_hungarian' # EP6
# os.makedirs(dir_exp, exist_ok=True)

# dir_data = '/labs/cliffordlab/data/EP6/openpose'
# dir_data = '/opt/pi-data/data_posenet' # EP6

# dir_proj_mat = dir_exp + '/proj_mat'
# dir_proj_mat = '/opt/pi-data/EP6_vision_server/camera_positions_realtime/proj_mat'

# dir_root = '/Users/hyeokalankwon/Documents/research_local/Emory_local/CEP/EP6' # hyeok local (MAC)

# hyeok local (Win)
dir_data = r'C:\Users\hyeok.kwon\Dropbox\research\EMORY\EP6/data/video' 
dir_exp =  r'D:\Research\Emory\EP6/exps/video'

use_ep6_grid = True

# file_ep6 = "/Users/jigardesai/Downloads/ep6_map_original.JPG" # pushti local
# file_ep6 = "../people_position/ep6_map_original.JPG" # oddjobs
# file_ep6 = dir_data + '/ep6_map_original.JPG'
# ep6_height = 451
# ep6_width = 708

if use_ep6_grid:
  file_ep6 = r'C:\Users\hyeok.kwon\Research\Emory\EP6' + '/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
  ep6_map = plt.imread(file_ep6)
  ep6_height, ep6_width, _ = ep6_map.shape

dir_proj_mat = dir_exp + '/proj_mat'

# pilot data collection
start_time = '20220318_161300' # yyyymmdd_HHMMSS
end_time =  '20220318_161400' # yyyymmdd_HHMMSS

start_time = datetime.strptime(start_time, '%Y%m%d_%H%M%S')
end_time = datetime.strptime(end_time, '%Y%m%d_%H%M%S')
assert end_time >= start_time, f'end_time({end_time}) should be equal or later then start_time({start_time}).'

delta_minute = timedelta(minutes=1) # per-minute
delta_second = timedelta(seconds=1) # per-minute

list_pi_ip = os.listdir(dir_proj_mat)
list_pi_ip.sort()
# pprint (list_pi_ip)

list_pi_ip = [item[6:9] for item in list_pi_ip]
# pprint (list_pi_ip)
# assert False

cmap = cm.get_cmap('rainbow')
c_list = cmap(np.linspace(0, 1, len(list_pi_ip)))
c_kps = cmap(np.linspace(0, 1, 10))

''' posenet '''
smooth_kps_len = 5 # sec

''' orientation '''
smooth_calb_len = 5 # sec

''' multi-view '''
# 1m: 10 (map) / 17.5 (grid)
overlap_th = 5 
overlap_th = 10
overlap_th = 15
overlap_th = 30

''' tracking '''
# 1m: 10 (map) / 17.5 (grid)
tracklet_th = 20 # 1m: 10
tracklet_th = 35 # 1m: 10
min_track_len = 5
smooth_track_len = 5

''' Grouping '''
# 1m: 10 (map) / 17.5 (grid)
interaction_th = 25
interaction_th = 43.5

''' Visualize '''
viz_frame = True
viz_kps = False
viz_kps_fp1 = False
viz_kps_fp2 = False
viz_kps_fn1 = False
viz_kps_fn2 = False
viz_kps_smth = False
viz_occup_indiv = False
viz_occup_mv = False
viz_occup_fn1 = False
viz_occup_fn2 = False
viz_tracking = True
viz_tracking_fp = True
viz_tracking_smth = True
viz_grouping = True

#---------------------------
# Per-pi camera occupancy

pi_data = {}

'''--- Load data first ---'''
# Let's not process within minute anymore
list_dt = list(timerange(start_time, end_time))
for t in range(len(list_dt)):
  dt_curr = list_dt[t]
  year = dt_curr.year
  month = dt_curr.month
  day = dt_curr.day
  hour = dt_curr.hour
  minute = dt_curr.minute
  second = dt_curr.second

  for pi_ip in list_pi_ip:
    pi = f'pi{pi_ip}.pi.bmi.emory.edu'
    
    dir_pi = dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
    dir_kps = dir_pi + '/keypoints'
    dir_vid = dir_pi + '/videos'

    file_kps = dir_kps + f'/{pi}{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}.npy'
    if not os.path.exists(file_kps):
      # print (file_kps, '... not exits')
      continue

    kps = np.load(file_kps)
    # print ('load from ...', file_kps)
    # print ('load from ...', file_kps, kps.shape)
        
    ''' collect frames '''
    if viz_frame:
      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      dir_copy_frame = dir_copy + '/frames'
      dir_copy_frame_pi = dir_copy_frame + f'/{pi}'
      os.makedirs(dir_copy_frame_pi, exist_ok=True)

      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      frame = plt.imread(file_frame)
      print ('load from ...', file_frame)
      frame = np.flip(frame, axis=1)

      file_frame_copy = dir_copy_frame_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.imsave(file_frame_copy, frame)
      print ('copy ...', file_frame, '... to ...', file_frame_copy)
    ''''''
                
    # remove zero keypoints
    kp_sum = np.sum(kps, axis=(1,2))
    kps = kps[kp_sum > 0,...]
    # print (kps.shape)
    if kps.shape[0] == 0: # remove empty keypoint
      continue
  
    # flip keypoints
    center_line = float(frame_width)/2
    kps[:,:,1] -= center_line
    kps[:,:,1] = -kps[:,:,1]
    kps[:,:,1] += center_line

    ''' draw kps on frames and save '''
    if viz_kps:
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      frame = plt.imread(file_frame)
      print ('load from ...', file_frame)
      frame = np.flip(frame, axis=1)

      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      dir_copy_kps = dir_copy + '/kps'
      dir_copy_kps_pi = dir_copy_kps + f'/{pi}'
      os.makedirs(dir_copy_kps_pi, exist_ok=True)

      fig, ax = plt.subplots()
      ax.imshow(frame)
      for i in range(kps.shape[0]):
        kp = kps[i]
        color = c_kps[i]

        for j1, j2 in connection:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color=color, markersize=2)
          ax.plot(x2, y2, 'o', color=color, markersize=2)
          ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
      
      file_copy_kps_pi = dir_copy_kps_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_kps_pi, bbox_inches='tight',pad_inches = 0)
      plt.close()
      print ('save in ...', file_copy_kps_pi)
    ''''''

    if pi not in pi_data:
      pi_data[pi] = {}

    key_dt = datetime(
      year=year,
      month=month,
      day=day,
      hour=hour,
      minute=minute,
      second=second)

    pi_data[pi][key_dt] = kps

if 0:
  for pi in pi_data:
    if 0:
      for key_dt in pi_data[pi]:
        print (pi, key_dt)
    else:
      print (pi, len(pi_data[pi].keys()))
    print ('-------------------')

if viz_frame | viz_kps:
  assert False
#-----------------------------------------------------

'''--- Pre-Process keypoints: Temporal Smoothing ---'''
''' 
Tracking on 2D image and ...
1) Remove "temporal" False Positives [v]
  1-1) remove non-changing poses from floor patterns
  1-2) remove no-matching kps in +/- 1 second frames
2) Remove False Negatives [v]
  2-1) for -1/0/1 second block,
       copy in non-tracked pose from +/- 1 to 0
  2-2) If 1 second is missing between frames,
       interpolate False Negatives
    
3) Smooth noisy poses (5 second window) [v]
'''

''' 1-1) Resolve Temporal False Positive '''
# remove non-changing poses from floor patterns
# not necessaryly between 1 second considering mising data.

dict_val_kps_fp_1 = {}
for pi in pi_data:
  dict_val_kps_fp_1[pi]= {}

  list_dt = list(pi_data[pi].keys())
  list_dt.sort()

  for t in range(len(list_dt)):
    dt_curr = list_dt[t]
    kps_curr = pi_data[pi][dt_curr]

    if t == len(list_dt)-1:
      dt_ref = list_dt[t-1]
    else:
      dt_ref = list_dt[t+1]
    kps_ref = pi_data[pi][dt_ref]

    val_curr = np.ones((kps_curr.shape[0],), dtype=bool)
    for i in range(kps_curr.shape[0]):
      kp_curr = kps_curr[i].reshape((1, 17, 2))
      diffs = np.sum(np.abs(kps_ref-kp_curr), axis=(1,2))
      if np.any(diffs == 0):
        val_curr[i] = False
    
    dict_val_kps_fp_1[pi][dt_curr] = val_curr

for pi in pi_data:
  for dt in pi_data[pi]:
    val_kps = dict_val_kps_fp_1[pi][dt]
    kps = pi_data[pi][dt]
    pi_data[pi][dt] = kps[val_kps]

# remove empty frame
pi_data_clean = {}
for pi in pi_data:
  for dt in pi_data[pi]:
    kps = pi_data[pi][dt]
    if kps.shape[0] > 0:
      if pi not in pi_data_clean:
        pi_data_clean[pi] = {}
      pi_data_clean[pi][dt] = kps
pi_data = pi_data_clean

''' draw kps on frames and save '''
if viz_kps_fp1:
  for pi in pi_data:
    for dt_curr in pi_data[pi]:
      year = dt_curr.year
      month = dt_curr.month
      day = dt_curr.day
      hour = dt_curr.hour
      minute = dt_curr.minute
      second = dt_curr.second

      dir_pi = dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      frame = plt.imread(file_frame)
      # print ('load from ...', file_frame)
      frame = np.flip(frame, axis=1)

      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      
      dir_copy_kps = dir_copy + '/kps_fp1'
      dir_copy_kps_pi = dir_copy_kps + f'/{pi}'
      os.makedirs(dir_copy_kps_pi, exist_ok=True)

      fig, ax = plt.subplots()
      ax.imshow(frame)

      kps = pi_data[pi][dt_curr]
      for i in range(kps.shape[0]):
        kp = kps[i]
        color = c_kps[i]

        for j1, j2 in connection:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color=color, markersize=2)
          ax.plot(x2, y2, 'o', color=color, markersize=2)
          ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
      
      file_copy_kps_pi = dir_copy_kps_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_kps_pi, bbox_inches='tight',pad_inches = 0)
      plt.close()
      print ('save in ...', file_copy_kps_pi)
  assert False
''''''
#-----------------------------------------------------

''' 1-2) Resolve Temporal False Positive '''
# find matching between second-1, second and second+1
# For a second, if no matching at all, then remove the pose
# ==> ACTUALLY, just consequtive windows ... so many cases that few sconds are missing

dict_val_kps_fp_2 = {}
for pi in pi_data:
  dict_val_kps_fp_2[pi] = {}

  list_dt = list(pi_data[pi].keys())
  list_dt.sort()

  for t in range(1, len(list_dt)-1):
    dt_curr = list_dt[t]
    kps_curr = pi_data[pi][dt_curr]
    val_curr = np.ones((kps_curr.shape[0],), dtype=bool)
    col_curr = None
    row_curr = None

    # prev & curr
    if 0:
      dt_prev = dt_curr - delta_second 
    else:
      # so many cases that few sconds are missing
      dt_prev = list_dt[t-1] 
    if dt_prev in pi_data[pi]:
      kps_prev = pi_data[pi][dt_prev]
      _, col_curr = kps_matching_between_frames(kps_prev, kps_curr)

    # curr & next
    if 0:
      dt_next = dt_curr + delta_second
    else:
      # so many cases that few sconds are missing
      dt_next = list_dt[t+1]
    if dt_next in pi_data[pi]:
      kps_next = pi_data[pi][dt_next]
      row_curr, _ = kps_matching_between_frames(kps_curr, kps_next)

    # print (pi)
    # print (dt_curr.second)
    # assert False
    if False and pi == 'pi148.pi.bmi.emory.edu' \
    and dt_curr.second == 55:
      row_curr, _ = kps_matching_between_frames(kps_curr, kps_next, verbose=True)
      assert False

    # both second +/- 1 does not exist.
    # then cannot decide -> just keep
    if col_curr is None \
    and row_curr is None:
      val_curr = np.ones((kps_curr.shape[0],), dtype=bool)
    else:
      for i in range(kps_curr.shape[0]):
        # second & second +/-1: middle
        if col_curr is not None \
        and row_curr is not None:
          if i not in col_curr \
          and i not in row_curr:
            val_curr[i] = 0

        # only second - 1: last
        elif col_curr is not None \
        and row_curr is None:
          if i not in col_curr:
            val_curr[i] = 0

        # only second + 1: first
        elif col_curr is None \
        and row_curr is not None:
          if i not in row_curr:
            val_curr[i] = 0
    
    dict_val_kps_fp_2[pi][dt_curr] = val_curr

for pi in pi_data:
  for dt in pi_data[pi]:
    if dt not in dict_val_kps_fp_2[pi]:
      continue
    val_kps = dict_val_kps_fp_2[pi][dt]
    kps = pi_data[pi][dt]
    pi_data[pi][dt] = kps[val_kps]

# remove empty frame
pi_data_clean = {}
for pi in pi_data:
  for dt in pi_data[pi]:
    kps = pi_data[pi][dt]
    if kps.shape[0] > 0:
      if pi not in pi_data_clean:
        pi_data_clean[pi] = {}
      pi_data_clean[pi][dt] = kps
pi_data = pi_data_clean


''' draw kps on frames and save '''
if viz_kps_fp2:
  for pi in pi_data:
    for dt_curr in pi_data[pi]:
      year = dt_curr.year
      month = dt_curr.month
      day = dt_curr.day
      hour = dt_curr.hour
      minute = dt_curr.minute
      second = dt_curr.second

      dir_pi = dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      frame = plt.imread(file_frame)
      # print ('load from ...', file_frame)
      frame = np.flip(frame, axis=1)

      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      
      dir_copy_kps = dir_copy + '/kps_fp2'
      dir_copy_kps_pi = dir_copy_kps + f'/{pi}'
      os.makedirs(dir_copy_kps_pi, exist_ok=True)

      fig, ax = plt.subplots()
      ax.imshow(frame)

      kps = pi_data[pi][dt_curr]
      for i in range(kps.shape[0]):
        kp = kps[i]
        color = c_kps[i]

        for j1, j2 in connection:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color=color, markersize=2)
          ax.plot(x2, y2, 'o', color=color, markersize=2)
          ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
      
      file_copy_kps_pi = dir_copy_kps_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_kps_pi, bbox_inches='tight',pad_inches = 0)
      plt.close('all')
      print ('save in ...', file_copy_kps_pi)
  assert False
''''''

''' 
----------------------------------------------------
So far Removed all possible False Positive cases.
If a pose is not captured/filtered/removed from a frame so far, we expdct that pose is also captured in other view, which can be recovered from multi-view approach.
Now, lets fill in for False Negatives
----------------------------------------------------
'''
#-----------------------------------------------------

''' 2-1) Resolve Temporal False Negative  '''
# 2nd order interpolation
# for -1/0/1 second block, copy in non-tracked pose from +/- 1 to 0
# ==> ACTUALLY, interpolate over captured timesteps... so may cases that few seconds are missing.
# <== Then, we will do 1st order interpolate to finely fill the missing frames each second in [2-2]

for pi in pi_data:
  list_dt = list(pi_data[pi].keys())
  list_dt.sort()
  for t in range(1, len(list_dt)-1):
    dt_curr = list_dt[t]
    dt_prev = list_dt[t-1]
    dt_next = list_dt[t+1]

    w = (dt_curr - dt_prev).seconds/(dt_next - dt_prev).seconds
    
    kps_curr = pi_data[pi][dt_curr]
    kps_prev = pi_data[pi][dt_prev]
    kps_next = pi_data[pi][dt_next]

    row_prev, col_curr = kps_matching_between_frames(kps_prev, kps_curr)

    row_curr, col_next = kps_matching_between_frames(kps_curr, kps_next)

    row_pprev, col_nnext = kps_matching_between_frames(kps_prev, kps_next)
    
    # if matching exist between -/+1 frames, but no matching with 0 frame, then it is false negative
    for r, c in zip(row_pprev, col_nnext):
      if r not in row_prev \
      and c not in col_next:
        kp_prev = kps_prev[r]
        kp_next = kps_next[c]
        kp_curr = (1-w)*kp_prev + w*kp_next
        kp_curr = kp_curr.reshape((1, 17, 2))
        kps_curr = np.concatenate((kps_curr, kp_curr))
    
    pi_data[pi][dt_curr] = kps_curr

# remove empty frame
pi_data_clean = {}
for pi in pi_data:
  for dt in pi_data[pi]:
    kps = pi_data[pi][dt]
    if kps.shape[0] > 0:
      if pi not in pi_data_clean:
        pi_data_clean[pi] = {}
      pi_data_clean[pi][dt] = kps
pi_data = pi_data_clean

''' draw kps on frames and save '''
if viz_kps_fn1:
  for pi in pi_data:
    for dt_curr in pi_data[pi]:
      year = dt_curr.year
      month = dt_curr.month
      day = dt_curr.day
      hour = dt_curr.hour
      minute = dt_curr.minute
      second = dt_curr.second

      dir_pi = dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      if os.path.exists(file_frame):
        frame = plt.imread(file_frame)
        # print (frame.shape)
        # assert False
        # print ('load from ...', file_frame)
        frame = np.flip(frame, axis=1)
      else:
        frame = np.ones((frame_height, frame_width, 3))

      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      
      dir_copy_kps = dir_copy + '/kps_fn1'
      dir_copy_kps_pi = dir_copy_kps + f'/{pi}'
      os.makedirs(dir_copy_kps_pi, exist_ok=True)

      fig, ax = plt.subplots()
      ax.imshow(frame)

      kps = pi_data[pi][dt_curr]
      for i in range(kps.shape[0]):
        kp = kps[i]
        color = c_kps[i]

        for j1, j2 in connection:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color=color, markersize=2)
          ax.plot(x2, y2, 'o', color=color, markersize=2)
          ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
      
      file_copy_kps_pi = dir_copy_kps_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_kps_pi, bbox_inches='tight',pad_inches = 0)
      plt.close('all')
      gc.collect()
      print ('save in ...', file_copy_kps_pi)
  assert False
''''''

#-----------------------------------------------------

''' 2-2) Resolve Temporal False Negative '''
# If 1 second is missing between frames, interpolate False Negatives
# ==> ACTUALLY, interpolation over few consequtive windows ... so many cases that few seconds are missing

for pi in pi_data:
  list_dt = list(pi_data[pi].keys())
  list_dt.sort()  
  for i in range(len(list_dt)-1):
    dt_prev = list_dt[i]
    dt_next = list_dt[i+1]
    interval = (dt_next - dt_prev).seconds
    # print (dt_prev, dt_next)
    # print (interval)
    # assert False
    if interval == 1:
      continue

    kps_prev = pi_data[pi][dt_prev]
    kps_next = pi_data[pi][dt_next]
    row_ind, col_ind = kps_matching_between_frames(kps_prev, kps_next)

    list_dt_interp = list(timerange(dt_prev, dt_next)) # dt_next not included
    for t in range(1, len(list_dt_interp)):
      w = t/(len(list_dt_interp))
      dt_curr = list_dt_interp[t]

      kps_curr = np.empty((0,17,2))

      used_prev = np.zeros((kps_prev.shape[0],))
      used_next = np.zeros((kps_next.shape[0],))
      for r, c in zip(row_ind, col_ind):
        kp_prev = kps_prev[r]
        kp_next = kps_next[c]
        kp_curr = (1-w)*kp_prev + w*kp_next
        kp_curr = kp_curr.reshape((1,17,2))
        kps_curr = np.concatenate((kps_curr, kp_curr))

        used_prev[r] = 1
        used_next[c] = 1

      if 0: # unreallistic that no movement at all for 1 sec.
        # for no matching just copy paste from prev and next
        for r in range(kps_prev.shape[0]):
          if not used_prev[r]:
            kp_curr = kps_prev[r]
            kps_curr = np.concatenate((kps_curr, kp_curr))
        
        for c in range(kps_next.shape[0]):
          if not used_next[c]:
            kp_curr = kps_next[c]
            kps_curr = np.concatenate((kps_curr, kp_curr))

      if kps_curr.shape[0] > 0:
        pi_data[pi][dt_curr] = kps_curr

# remove empty frame
pi_data_clean = {}
for pi in pi_data:
  for dt in pi_data[pi]:
    kps = pi_data[pi][dt]
    if kps.shape[0] > 0:
      if pi not in pi_data_clean:
        pi_data_clean[pi] = {}
      pi_data_clean[pi][dt] = kps
pi_data = pi_data_clean

''' draw kps on frames and save '''
if viz_kps_fn2:
  for pi in pi_data:
    for dt_curr in pi_data[pi]:
      year = dt_curr.year
      month = dt_curr.month
      day = dt_curr.day
      hour = dt_curr.hour
      minute = dt_curr.minute
      second = dt_curr.second

      dir_pi = dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      if os.path.exists(file_frame):
        frame = plt.imread(file_frame)
        # print (frame.shape)
        # assert False
        # print ('load from ...', file_frame)
        frame = np.flip(frame, axis=1)
      else:
        frame = np.ones((frame_height, frame_width, 3))

      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      
      dir_copy_kps = dir_copy + '/kps_fn2'
      dir_copy_kps_pi = dir_copy_kps + f'/{pi}'
      os.makedirs(dir_copy_kps_pi, exist_ok=True)

      fig, ax = plt.subplots()
      ax.imshow(frame)

      kps = pi_data[pi][dt_curr]
      for i in range(kps.shape[0]):
        kp = kps[i]
        color = c_kps[i]

        for j1, j2 in connection:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color=color, markersize=2)
          ax.plot(x2, y2, 'o', color=color, markersize=2)
          ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
      
      file_copy_kps_pi = dir_copy_kps_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_kps_pi, bbox_inches='tight',pad_inches = 0)
      plt.close('all')
      gc.collect()
      print ('save in ...', file_copy_kps_pi)
  assert False
''''''

''' 
----------------------------------------------------
So far Removed all possible False Positive & Negative cases.
Now, lets clean the noisy poses.
----------------------------------------------------
'''

#-----------------------------------------------------
''' 3) Smooth noisy poses (5 second window) '''
# find consecutive seconds. (https://stackoverflow.com/a/2361991) and smooth keypoints.

pi_data_clean = {}
for pi in pi_data:
  pi_data_clean[pi] = {}

  list_dt = list(pi_data[pi].keys())
  list_dt.sort()
  list_ts = [datetime.timestamp(item) for item in list_dt]

  for k, g in groupby(enumerate(list_ts), lambda ix: ix[0]-ix[1]):
    block_ts = list(map(itemgetter(1), g))
    block_dt = [datetime.fromtimestamp(item) for item in block_ts]
    # print (pi)
    # pprint (block_dt)
    # print ('----------------')
    # continue

    # track keypoints in the block
    pid_seq = {}
    tracker_kps = hungarian_kps()
    for i in range(len(block_dt)):
      dt_curr = block_dt[i]
      kps_curr = pi_data[pi][dt_curr]

      try:
        pid = tracker_kps.update(kps_curr)
      except:
        print (pi, dt_curr)
        print (kps_curr)
        assert False

      for rid, pid in enumerate(pid):
        if pid not in pid_seq:
          pid_seq[pid] = {'dt': [], 'rid': []}
        pid_seq[pid]['dt'].append(dt_curr)
        pid_seq[pid]['rid'].append(rid)
      
    # for pid in pid_seq:
    #   print (pid)
    #   pprint (pid_seq[pid]['dt'])
    #   print ('---')
    # print ('--------------------')
    
    # collect cleaned keypoints
    for pid in pid_seq:
      dts = pid_seq[pid]['dt']
      rids = pid_seq[pid]['rid']
      if len(dts) > 1:
        # collect sequence
        kps_seq = np.empty((0, 17*2))
        for dt, rid in zip(dts, rids):
          kps = pi_data[pi][dt][rid].reshape((1, 17*2))
          kps_seq = np.concatenate((kps_seq, kps))
        
        # smoothe
        for ch in range(kps_seq.shape[1]):
          kps_seq_ch = kps_seq[:, ch]
          kps_seq_ch = Series(kps_seq_ch).rolling(smooth_kps_len, min_periods=1, center=True).mean().to_numpy()
          kps_seq[:, ch] = kps_seq_ch
        
        kps_seq = kps_seq.reshape((-1, 17, 2))

        for i, dt in enumerate(dts):
          if dt not in pi_data_clean[pi]:
            pi_data_clean[pi][dt] = np.empty((0, 17, 2))

          kps_clean = kps_seq[i].reshape((1, 17, 2))
          
          kps_clean_dt = pi_data_clean[pi][dt]
          kps_clean_dt = np.concatenate((kps_clean_dt, kps_clean))
          pi_data_clean[pi][dt] = kps_clean_dt
          
pi_data = pi_data_clean

''' draw kps after false positive removal on frames and save '''
if viz_kps_smth:
  for pi in pi_data:
    for dt in pi_data[pi]:
      kps = pi_data[pi][dt]
      year = dt.year
      month = dt.month
      day = dt.day
      hour = dt.hour
      minute = dt.minute
      second = dt.second

      dir_pi = dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      if os.path.exists(file_frame):
        frame = plt.imread(file_frame)
        # print (frame.shape)
        # assert False
        # print ('load from ...', file_frame)
        frame = np.flip(frame, axis=1)
      else:
        frame = np.ones((frame_height, frame_width, 3))

      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      
      dir_copy_kps_fp_rm = dir_copy + '/kps_smth'
      dir_copy_kps_fp_rm_pi = dir_copy_kps_fp_rm + f'/{pi}'
      os.makedirs(dir_copy_kps_fp_rm_pi, exist_ok=True)

      fig, ax = plt.subplots()
      ax.imshow(frame)
      for i in range(kps.shape[0]):
        kp = kps[i]
        color = c_kps[i]

        for j1, j2 in connection:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color=color, markersize=2)
          ax.plot(x2, y2, 'o', color=color, markersize=2)
          ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
      
      file_copy_kps_fp_rm_pi = dir_copy_kps_fp_rm_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
      plt.close()
      print ('save in ...', file_copy_kps_fp_rm_pi)
  assert False
''''''


'''
-----------------------------------------------
All the process for raw keypoints are finished.
Now move on to Occupancy Analysis
-----------------------------------------------
'''


''' 
Start Occupancy analysis
Get EP6 occupancy from average feet position 
'''
occup_pi = {}
for pi in pi_data:
  pi_ip = pi[2:5]
  file_save = dir_proj_mat +f'/pi_ip_{pi_ip}.npy'
  assert os.path.exists(file_save), pi_ip
  if not os.path.exists(file_save):
    print (file_save, '... not exist')
    continue
  M = np.load(file_save)
  # print ('load from ...', file_save)

  for dt in pi_data[pi]:
    if dt not in occup_pi:
      occup_pi[dt] = {}

    kps = pi_data[pi][dt]
    avg_feet_positions = extract_avg_feet_positions(kps)
    if avg_feet_positions.shape[0] == 0:
      continue

    src = np.array([[y, x, 1] for [x, y, z] in avg_feet_positions], dtype='float32').T
    # print (src.shape)

    # project EP6
    EP6_feet_pos = M.dot(src)
    # print (EP6_feet_pos)
    EP6_feet_pos /= EP6_feet_pos[2,:].reshape((1,-1))
    # print (EP6_feet_pos)
    EP6_feet_pos = EP6_feet_pos[:2].T
    # print (EP6_feet_pos)

    # remove outliers
    EP6_feet_pos[:,0] = np.clip(EP6_feet_pos[:,0], 0, ep6_width-1)
    EP6_feet_pos[:,1] = np.clip(EP6_feet_pos[:,1], 0, ep6_height-1)
    occup_pi[dt][pi] = EP6_feet_pos
      
    ''' draw occupancy '''
    if viz_occup_indiv:
      year = dt.year
      month = dt.month
      day = dt.day
      hour = dt.hour
      minute = dt.minute
      second = dt.second
      
      dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if use_ep6_grid:
        dir_copy += '_grid'
      
      dir_copy_occup = dir_copy + '/occup'
      dir_copy_occup_pi = dir_copy_occup + f'/{pi}'
      os.makedirs(dir_copy_occup_pi, exist_ok=True)    

      fig, ax = plt.subplots()
      ax.imshow(ep6_map)
      for i in range(EP6_feet_pos.shape[0]):
        color = c_kps[i]
        occup_ = EP6_feet_pos[i]

        ax.scatter(occup_[0], occup_[1], s=5, color=color, marker='^', alpha=0.7)

      file_copy_occup_pi = dir_copy_occup_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
      plt.close()
      print ('save in ...', file_copy_occup_pi)
      ''''''
# assert False

if viz_occup_indiv:
  ''' draw occupancy observed from all Pis '''
  list_pi = list(pi_data.keys())
  list_pi.sort()
  c_pis = cmap(np.linspace(0, 1, len(list_pi)))
  pi_color = {}
  for pi, c in zip(list_pi, c_pis):
    pi_color[pi] = c

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_occup = dir_copy + '/occup'
  dir_copy_occup_indiv = dir_copy_occup + '/individual'
  os.makedirs(dir_copy_occup_indiv, exist_ok=True)

  for dt in occup_pi:
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    timestep = f'{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02d}'
    
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for pi in occup_pi[dt]:
          
      occup = occup_pi[dt][pi]
      color = pi_color[pi]

      ax.scatter(occup[:,0], occup[:,1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

    ax.legend()

    file_copy_occup_pi = dir_copy_occup_indiv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  assert False

''' Multi-view: Person matching and association across camera '''

list_pi = list(pi_data.keys())
list_pi.sort()

occup_mv = {}
for dt in occup_pi:
  year = dt.year
  month = dt.month
  day = dt.day
  hour = dt.hour
  minute = dt.minute
  second = dt.second
  timestep = f'{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02d}'

  G = nx.Graph()
  for pi in occup_pi[dt]:
    occup = occup_pi[dt][pi]
    for i in range(occup.shape[0]):
      loc_i = occup[i]
      node = f'{pi}_{i}'
      if node not in G:
        G.add_node(node, loc=loc_i)    
  
  for i in range(len(list_pi)-1):
    for j in range(i+1, len(list_pi)):
      pi_i = list_pi[i]
      pi_j = list_pi[j]

      if pi_i not in occup_pi[dt] \
      or pi_j not in occup_pi[dt]:
        continue

      occup_i = occup_pi[dt][pi_i]
      occup_j = occup_pi[dt][pi_j]

      dist_mat = np.empty((occup_i.shape[0], occup_j.shape[0]))
      dist_mat[:] = np.inf
      for r in range(occup_i.shape[0]):
        for c in range(occup_j.shape[0]):
          loc_m = occup_i[r]
          loc_n = occup_j[c]

          node1 = f'{pi_i}_{r}'
          if node1 not in G:
            G.add_node(node1, loc=loc_m)

          node2 = f'{pi_j}_{c}'
          if node2 not in G:
            G.add_node(node2, loc=loc_n)

          dist = np.sqrt(np.sum((loc_m - loc_n)**2))
          dist_mat[r,c] = dist

      row_ind, col_ind = linear_sum_assignment(dist_mat)

      for r,c in zip(row_ind, col_ind):
        dist = dist_mat[r, c]
      
        if dist < overlap_th:
          node1 = f'{pi_i}_{r}'
          node2 = f'{pi_j}_{c}'
          # print (node1, node2)
          G.add_edge(node1, node2)
        
  if 0:
    # find connected components
    for cc in nx.connected_components(G):
      # check if connected component include multiple occups from a pi
      # this should not happen
      # if so, then remove edge
      if 1:
        list_cc_pi = []
        for pi_o in cc:
          pi = pi_o.split('_')[0]
          if pi not in list_cc_pi:
            list_cc_pi.append(pi)
          else:
            print (cc)
            assert False
      print (cc, len(cc))
    print ('---------------------')   

  occup_mv[dt] = []
  for cc in nx.connected_components(G):
    cc = list(cc)
    # print (cc)

    n_node = 0
    occup = np.zeros((2,))
    for pi_o in cc:
      pi, i = cc[0].split('_')
      i = int(i)
      _occup = occup_pi[dt][pi][i]
      # print (occup.shape)
      # assert False
      occup += _occup
      n_node += 1

    occup /= n_node
    occup_mv[dt].append(occup)

  occup_mv[dt] = np.array(occup_mv[dt])

  if viz_occup_mv:
    ''' draw connected components from occupancy observed from all Pis '''
    list_pi = list(pi_data.keys())
    list_pi.sort()
    c_pis = cmap(np.linspace(0, 1, len(list_pi)))
    pi_color = {}
    for pi, c in zip(list_pi, c_pis):
      pi_color[pi] = c

    dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if use_ep6_grid:
      dir_copy += '_grid'
    
    dir_copy_occup = dir_copy + '/occup'
    dir_copy_occup_cc = dir_copy_occup + f'/connected_component_ov_{overlap_th}'
    os.makedirs(dir_copy_occup_cc, exist_ok=True)

    # pprint (list(G.nodes(data=True)))
    # print (len(G.nodes(data=True)))
    # for node in G.nodes(data=True):
    #   print (node)
    #   print (node[0])
    #   print (node[1])
    #   print (node[1]['loc'])
    #   print ('--------')
    # print ('========================')
    # pprint (list(G.edges(data=True)))
    # print (len(G.edges(data=True)))
    # assert False

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    for node in G.nodes(data=True):
      pi = node[0].split('_')[0]
      pi_ip = pi[2:5]
      loc = node[1]['loc']
      color = pi_color[pi]
      ax.scatter(loc[0], loc[1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

    # for pi in occup_pi[dt]:
    #   occup = occup_pi[dt][pi]
    #   color = pi_color[pi]
    #   pi_ip = pi[2:5]
    #   ax.scatter(occup[:,0], occup[:,1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

    if len(G.edges(data=True)) > 0:    
      # pprint (list(G.nodes(data=True)))
      # print (len(G.edges(data=True)))
      # print (len(G.edges()))
      # print (list(G.edges(data=True)))
      # print (list(G.edges()))
      # print ('=========')
      # for edge in G.edges():
      #   print (edge)
      #   node1, node2 = edge
      #   print (node1)
      #   print (G.nodes[node1])
      #   print (G.nodes[node1]['loc'])
      #   print (node2)
      #   print (G.nodes[node2])
      #   print (G.nodes[node2]['loc'])
      #   print ('-----')
      # assert False

      for edge in G.edges():
        n1, n2 = edge
        loc_m = G.nodes[n1]['loc']
        loc_n = G.nodes[n2]['loc']

        x = [loc_m[0], loc_n[0]]
        y = [loc_m[1], loc_n[1]]
    
        ax.plot(x, y, c='k')

    ax.legend()

    file_copy_occup_pi = dir_copy_occup_cc + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

# assert False

if viz_occup_mv:
  ''' draw occupancy with multi-view '''
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_occup = dir_copy + '/occup'
  dir_copy_occup_mv = dir_copy_occup + f'/multi-view_ov_{overlap_th}'
  os.makedirs(dir_copy_occup_mv, exist_ok=True)

  for dt in occup_mv:
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    occup = occup_mv[dt]

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

    file_copy_occup_pi = dir_copy_occup_mv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  
  assert False

''' Save Multi-view occupancy  '''
# pushti's code
# dir_occup = dir_exp + '/occupancy'
# dir_save = dir_occup + f'/{year}.{month}.{day}_{hour}.{minute}'
# dir_save = '/Users/jigardesai/Downloads/test'

# hyeok's code
dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
if use_ep6_grid:
  dir_copy += '_grid'

dir_save = dir_copy + '/occup'

os.makedirs(dir_save, exist_ok=True)

file_save = dir_save + '/multi-view.p'
cp.dump(occup_mv, open(file_save, 'wb'))
print ('save in ...', file_save)

#-------------------------------------------------

''' Post-Process Multi-view: Temporal Smoothing
1) Remove False Negatives: 2nd order interpolation
2) Remove False Negatives: 1st order interpolation
'''

''' 1) Remove False Negatives: 2nd order interpolation  '''
list_dt = list(occup_mv.keys())
list_dt.sort()
for t in range(1, len(list_dt)-1):
  dt_curr = list_dt[t]
  dt_prev = list_dt[t-1]
  dt_next = list_dt[t+1]

  w = (dt_curr - dt_prev).seconds/(dt_next - dt_prev).seconds

  occup_curr = occup_mv[dt_curr]
  occup_prev = occup_mv[dt_prev]
  occup_next = occup_mv[dt_next]

  row_prev, col_curr = occup_matching_between_frames(occup_prev, occup_curr, th=overlap_th)

  row_curr, col_next = occup_matching_between_frames(occup_curr, occup_next, th=overlap_th)

  row_pprev, col_nnext = occup_matching_between_frames(occup_prev, occup_next, th=overlap_th)

  # if matching exist between -/+1 frames, but no matching with 0 frame, then it is false negative
  for r, c in zip(row_pprev, col_nnext):
    if r not in row_prev \
    and c not in col_next:
      oc_prev = occup_prev[r]
      oc_next = occup_next[c]
      oc_curr = (1-w)*oc_prev + w*oc_next
      oc_curr = oc_curr.reshape((1, 2))
      occup_curr = np.concatenate((occup_curr, oc_curr))
  occup_mv[dt_curr] = occup_curr  

# remove empty frame
occup_mv_clean = {}
for dt in occup_mv:
  occup = occup_mv[dt]
  if occup.shape[0] > 0:
    occup_mv_clean[dt] = occup
occup_mv = occup_mv_clean

if viz_occup_fn1:
  ''' draw occupancy with multi-view '''
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_occup = dir_copy + '/occup'
  dir_copy_occup_mv = dir_copy_occup + f'/multi-view_ov_{overlap_th}_fn1'
  os.makedirs(dir_copy_occup_mv, exist_ok=True)

  for dt in occup_mv:
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    occup = occup_mv[dt]

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

    file_copy_occup_pi = dir_copy_occup_mv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  
  assert False

#-------------------------------------------------


''' 2) Remove False Negatives: 1st order interpolation '''
list_dt = list(occup_mv.keys())
list_dt.sort()
for t in range(len(list_dt)-1):
  dt_prev = list_dt[i]
  dt_next = list_dt[i+1]
  interval = (dt_next - dt_prev).seconds
  if interval == 1:
    continue

  occup_prev = occup_mv[dt_prev]
  occup_next = occup_mv[dt_next]
  row_int, col_ind = occup_matching_between_frames(occup_prev, occup_next, th=overlap_th)

  list_dt_interp = list(timerange(dt_prev, dt_next)) # dt_next not included
  for t in range(1, len(list_dt_interp)):
    w = t/(len(list_dt_interp))
    dt_curr = list_dt_interp[t]

    occup_curr = np.empty((0,2))
    for r,c in zip(row_int, col_ind):
      oc_prev = occup_prev[r]
      oc_next = occup_next[c]
      oc_curr = (1-w)*oc_prev + w*oc_next
      oc_curr = oc_curr.reshape((1,2))
      occup_curr = np.concatenate(occup_curr, oc_curr)
    
    if occup_curr.shape[0] > 0:
      occup_mv[dt_curr] = occup_curr

# remove empty frame
occup_mv_clean = {}
for dt in occup_mv:
  occup = occup_mv[dt]
  if occup.shape[0] > 0:
    occup_mv_clean[dt] = occup
occup_mv = occup_mv_clean

if viz_occup_fn2:
  ''' draw occupancy with multi-view '''
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_occup = dir_copy + '/occup'
  dir_copy_occup_mv = dir_copy_occup + f'/multi-view_ov_{overlap_th}_fn2'
  os.makedirs(dir_copy_occup_mv, exist_ok=True)

  for dt in occup_mv:
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    occup = occup_mv[dt]

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

    file_copy_occup_pi = dir_copy_occup_mv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  
  assert False

#--------------------------------------------

''' Tracking with Hungarian Method '''
list_dt = list(occup_mv.keys())
list_dt.sort()
list_ts = [datetime.timestamp(item) for item in list_dt]

mv_seq = {}
max_pid = None
for k, g in groupby(enumerate(list_ts), lambda ix: ix[0]-ix[1]):
  block_ts = list(map(itemgetter(1), g))
  block_dt = [datetime.fromtimestamp(item) for item in block_ts]
  # print (len(block_dt))

  tracker_occup = hungarian_occup(max_pid=max_pid,
                                  th=tracklet_th)
  for i in range(len(block_dt)):
    dt_curr = block_dt[i]
    occup_curr = occup_mv[dt_curr]
    pid = tracker_occup.update(occup_curr)
    for rid, pid  in enumerate(pid):
      if pid not in mv_seq:
        mv_seq[pid] = {
          'dt': [],
          'rid': [],
          'loc': []}
      if max_pid is None:
        max_pid = pid
      elif max_pid < pid:
        max_pid = pid
      # print (max_pid, pid)

      mv_seq[pid]['dt'].append(dt_curr)
      mv_seq[pid]['rid'].append(rid)
      mv_seq[pid]['loc'].append(occup_curr[rid])

print ('--------')
for pid in mv_seq:
  print (pid, len(mv_seq[pid]['loc']))
print ('--------')
# assert False

if viz_tracking | viz_tracking_fp | viz_tracking_smth | viz_grouping:
  ''' plot all tracks on a single figure '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}'
  
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

  file_save = dir_save + f'/all_Tracks.png'
  plt.savefig(file_save)
  plt.close()  
  print ('save in ...', file_save)
  # assert False

  ''' draw tracking for each person '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    file_save = dir_save + f'/id_{int(pid):03d}.png'
    plt.savefig(file_save)
    plt.close()  
    print ('save in ...', file_save)
  # assert False

  ''' draw for each second for resulting tracks '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color
  
  # each second
  mv_seq_dt = {}
  for pid in mv_seq:
    for dt_curr, rid, loc in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)

  for dt_curr in mv_seq_dt:
    year = dt_curr.year
    month = dt_curr.month
    day = dt_curr.day
    hour = dt_curr.hour
    minute = dt_curr.minute
    second = dt_curr.second

    dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if use_ep6_grid:
      dir_copy += '_grid'
    
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}'
    
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc']):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]

      ax.plot(xs, ys, c='k')
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)
  
  # assert False

#--------------------------------------------

''' Post-Process tracking:
Remove False Positives: Remove short tracklets (<= 5 sec)
'''
mv_seq_cleaned = {}
for pid in mv_seq:
  if len(mv_seq[pid]['loc']) > min_track_len:
    mv_seq_cleaned[pid] = mv_seq[pid]
    # print (pid, len(mv_seq[pid]['loc']))
    # print (mv_seq_cleaned[pid]['loc'])
    # print (mv_seq[pid]['loc'])
    # print ('----------------------')
mv_seq = mv_seq_cleaned
# assert False

print ('--------')
for pid in mv_seq:
  print (pid, len(mv_seq[pid]['loc']))
print ('--------')
# assert False

if viz_tracking | viz_tracking_fp | viz_tracking_smth | viz_grouping:
  ''' plot all tracks on a single figure '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_fp'
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

  file_save = dir_save + f'/all_Tracks.png'
  plt.savefig(file_save)
  plt.close()  
  print ('save in ...', file_save)
  # assert False
  
  ''' draw tracking for each person '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_fp'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    file_save = dir_save + f'/id_{int(pid):03d}.png'
    plt.savefig(file_save)
    plt.close('all')  
    print ('save in ...', file_save)
  # assert False
  
  ''' draw for each second for resulting tracks '''
  # each second
  mv_seq_dt = {}
  for pid in mv_seq:
    for dt_curr, rid, loc in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)

  for dt_curr in mv_seq_dt:
    year = dt_curr.year
    month = dt_curr.month
    day = dt_curr.day
    hour = dt_curr.hour
    minute = dt_curr.minute
    second = dt_curr.second

    dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if use_ep6_grid:
      dir_copy += '_grid'
    
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_fp'
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc']):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]
      ax.plot(xs, ys, c='k', label=pid)
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)
  
  # assert False

#--------------------------------------------

''' Post-Process tracking: Temporal Smoothing
Temporal Smoothing
'''
for pid in mv_seq:
  locs = np.array(mv_seq[pid]['loc'])

  # smoothe
  for ch in range(locs.shape[1]):
    locs_ch = locs[:, ch]
    locs_ch = Series(locs_ch).rolling(smooth_track_len, min_periods=1, center=True).mean().to_numpy()
    locs[:, ch] = locs_ch

  mv_seq[pid]['loc'] = locs

if viz_tracking | viz_tracking_fp | viz_tracking_smth | viz_grouping:
  ''' plot all tracks on a single figure '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_smth'
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

  file_save = dir_save + f'/all_Tracks.png'
  plt.savefig(file_save)
  plt.close()  
  print ('save in ...', file_save)
  # assert False
  
  ''' draw tracking for each person '''
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_smth'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    file_save = dir_save + f'/id_{int(pid):03d}.png'
    plt.savefig(file_save)
    plt.close('all')  
    print ('save in ...', file_save)
  # assert False
  
  ''' draw for each second for resulting tracks '''
  # each second
  mv_seq_dt = {}
  for pid in mv_seq:
    for dt_curr, rid, loc in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)

  for dt_curr in mv_seq_dt:
    year = dt_curr.year
    month = dt_curr.month
    day = dt_curr.day
    hour = dt_curr.hour
    minute = dt_curr.minute
    second = dt_curr.second

    dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if use_ep6_grid:
      dir_copy += '_grid'
    
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_smth'
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc']):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]
      ax.plot(xs, ys, c=color, label=pid)
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)

#--------------------------------------------

'''
Group detection
Graph-based approach: 
Connected components are groups.
Clque is very conservative approach.
'''

# Get per frame occupancy from the cleaned trajectory
occup_track_cleaned = {}
for pid in mv_seq:
  locs = mv_seq[pid]['loc']
  dts = mv_seq[pid]['dt']

  for dt, loc in zip(dts, locs):
    if dt not in occup_track_cleaned:
      occup_track_cleaned[dt] = np.empty((0,2))
    
    occup = occup_track_cleaned[dt]
    loc = loc.reshape((1,2))
    occup = np.concatenate((occup, loc))
    occup_track_cleaned[dt] = occup

# Find one-on-one interaction
interactions = {}
for dt in occup_track_cleaned:

  occup = occup_track_cleaned[dt]

  G = nx.Graph()
  for i in range(occup.shape[0]):
    G.add_node(i)
  
  for i in range(occup.shape[0]-1):
    for j in range(i+1, occup.shape[0]):
      loc_i = occup[i]
      loc_j = occup[j]
      dist = np.sqrt(np.sum((loc_i - loc_j)**2))

      if dist < interaction_th:
        G.add_edge(i, j)
  
  interactions[dt] = []
  # for cc in nx.enumerate_all_cliques(G):
  for cc in nx.connected_components(G):
    cc = list(cc)
    # print (cc)
    # assert False

    if len(cc) > 1:
      interactions[dt].append(cc)
  
if viz_tracking | viz_tracking_fp | viz_tracking_smth | viz_grouping:
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  dir_copy_group = dir_copy + f'/group_th_{interaction_th}'
  os.makedirs(dir_copy_group, exist_ok=True)

  for dt in occup_track_cleaned:
    # print (dt)
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    occup = occup_track_cleaned[dt]

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7) 

    for grp in interactions[dt]:
      # print (grp)
      for i in range(len(grp)-1):
        idx1 = grp[i]
        idx2 = grp[i+1]
        # print (idx1, idx2)
        loc1 = occup[idx1]
        loc2 = occup[idx2]

        x = [loc1[0], loc2[0]]
        y = [loc1[1], loc2[1]]
        # print (x, y)
    
        ax.plot(x, y, c='k')
        
    file_copy_occup_pi = dir_copy_group + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
