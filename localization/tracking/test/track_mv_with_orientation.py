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
from utils.grouping import detect_direct_interaction

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

''' tracking & orientation'''
w_tr = 0.7 # moving direction is more important

''' Grouping '''
# 1m: 10 (map) / 17.5 (grid)
interaction_th = 25
interaction_th = 43.5
interaction_ori = 180 # 180 degrees

''' Visualize '''
viz_frame = 0

viz_kps = 0
viz_kps_fp1 = 0
viz_kps_fp2 = 0
viz_kps_fn1 = 0
viz_kps_fn2 = 0
viz_kps_smth = 0

via_kps_track = 0
viz_pose3d = 0
viz_calib_proj = 0
viz_extrinsic = 0
viz_extrinsic_smth = 0
viz_pose3d_calib = 0
viz_pose3d_calib_match_2d = 0
viz_chest_3d = 0
viz_chest_2d = 0
viz_chest_2d_cam = 0
viz_chest_2d_ep6 = 0
viz_occup_ori_ep6_pi = 0

viz_occup_ori = 1

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
      dir_copy += '_ori'
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
      dir_copy += '_ori'
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
      dir_copy += '_ori'
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
      dir_copy += '_ori'
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
      dir_copy += '_ori'
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
      dir_copy += '_ori'
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
      dir_copy += '_ori'
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
Now move on to 1) Orientation & 2) Occupancy Analysis
-----------------------------------------------
'''

'''
1) Orientation Estimation
- a. Generate 3D pose
- b. PnP 2D & 3D pose
- c. smooth calibrated orientations
- d. calibrate 3D pose
- e. Get 3D chest normal vector
- f. Project on the x-y plane
- g. Reflect camera direction in EP6
- h. Project camera-reflected orientation to EP6 axis
- i. project foot location
'''

# prepare 3D lifting model
sys.path.append(r'C:\Users\hyeok.kwon\Research\Emory\EP6\repos\VideoPose3D')

coco_metadata = {
		'layout_name': 'coco',
		'num_joints': 17,
		'keypoints_symmetry': [
				# 0: nose
				[1, # left eye
				3, # left ear
				5, # left shoulder
				7,  # left elbow
				9,  # left wrist
				11, # left hip
				13, # left knee
				15], # left ankle
				[2, # right eye
				4, # right ear
				6, # right shoulder
				8, # right elbow
				10, # right wrist
				12, # right hip
				14, # right knee
				16], # right ankle
		]
}

metadata = coco_metadata
metadata['video_metadata'] = {'temp_name': {'h': frame_height, 'w': frame_width}}
keypoints_symmetry = metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
from skeleton.h36m import Human36mDataset, Skeleton
h36m = Human36mDataset()
skeleton = h36m.skeleton()
joints_left, joints_right = list(h36m.skeleton().joints_left()), list(h36m.skeleton().joints_right())

from common.model import TemporalModel
architecture = '3,3,3,3,3'
filter_widths = [int(x) for x in architecture.split(',')]
causal = False
dropout = 0.25
channels = 1024
dense = False
model_pos = TemporalModel(
	metadata['num_joints'], 
	2, 
	metadata['num_joints'],
	filter_widths=filter_widths, 
	causal=causal, 
	dropout=dropout, 
	channels=channels,
	dense=dense)

receptive_field = model_pos.receptive_field()
# print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if causal:
		# print('INFO: Using causal convolutions')
		causal_shift = pad
else:
		causal_shift = 0

assert torch.cuda.is_available()
device = torch.device('cuda')
model_pos = model_pos.to(device)

chk_filename = r'D:\Research\Emory\EP6\repos\VideoPose3D\model\pretrained_h36m_detectron_coco.bin'
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
epoch = checkpoint['epoch']
print(f'This model was trained for {epoch} epochs')
model_pos.load_state_dict(checkpoint['model_pos'])

from common.camera import normalize_screen_coordinates
from scipy.spatial.transform import Rotation as R

# prepare camera matrices
size = [frame_height, frame_width]
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], 
                        dtype = "double")
dist_coeffs = np.zeros((4,1)) # no lens distortion

# chest keypoints
#     {12, "RHip"},
#     {11, "LHip"},
#     {5,  "LShoulder"},
#     {6,  "RShoulder"},
idx_2d = np.array([12,11,5,6])
# 'RightHip': 1,          # Hip.R
# 'LeftHip': 4,           # Hip.L
# 'LeftShoulder': 11,     # Shoulder.L
# 'RightShoulder': 14,    # Shoulder.R
idx_3d = np.array([1,4,11,14])

solvepnp_flag = cv2.SOLVEPNP_EPNP # So this gives the best improvment

#--------------------------

ori_pi = {}
occup_pi = {}
for pi in pi_data:

  pi_ip = int(pi[2:5])
  file_save = dir_proj_mat +f'/pi_ip_{pi_ip}.npy'
  assert os.path.exists(file_save), pi_ip
  if not os.path.exists(file_save):
    print (file_save, '... not exist')
    continue
  M = np.load(file_save)
  
  list_dt = list(pi_data[pi].keys())
  list_dt.sort()
  list_ts = [datetime.timestamp(item) for item in list_dt]

  max_pid = None
  for k, g in groupby(enumerate(list_ts), lambda ix: ix[0]-ix[1]):
    block_ts = list(map(itemgetter(1), g))
    block_dt = [datetime.fromtimestamp(item) for item in block_ts]

    # track keypoints in the block
    pid_seq = {}
    tracker_kps = hungarian_kps(max_pid=max_pid)
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
        if max_pid is None:
          max_pid = pid
        elif max_pid < pid:
          max_pid = pid
          
        pid_seq[pid]['dt'].append(dt_curr)
        pid_seq[pid]['rid'].append(rid)
    
    for pid in pid_seq:
      dts = pid_seq[pid]['dt']
      rids = pid_seq[pid]['rid']
      if len(dts)> 1:

        # collect sequence
        kps_seq = np.empty((0, 17, 2))
        for dt, rid in zip(dts, rids):
          kps = pi_data[pi][dt][rid].reshape((1, 17, 2))
          kps_seq = np.concatenate((kps_seq, kps))
        
        if via_kps_track:
          for i, dt in enumerate(dts):
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
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/kps_track'
            os.makedirs(dir_copy_pid, exist_ok=True)

            fig, ax = plt.subplots()
            ax.imshow(frame)

            kp = kps_seq[i]
            color = 'k'

            for j1, j2 in connection:
              # print (j1, j2)
              # print (kp[j1-1], kp[j2-1])

              y1, x1 = kp[j1-1]
              y2, x2 = kp[j2-1]

              ax.plot(x1, y1, 'o', color=color, markersize=2)
              ax.plot(x2, y2, 'o', color=color, markersize=2)
              ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)

            # left shoulder -> Actually right shoulder (video flip error)
            y1, x1 = kp[5]
            ax.plot(x1, y1, 'o', color='r', markersize=5)
            # right shoulder -> Actually left shoulder (video flip error)
            y1, x1 = kp[6]
            ax.plot(x1, y1, 'o', color='b', markersize=5)              
            
            file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_copy_kps_fp_rm_pi)
        
        ''' a. lift 2D -> 3D '''
        kps_seq_norm = copy.deepcopy(kps_seq)
        kps_seq_norm[..., :2] = normalize_screen_coordinates(kps_seq_norm[..., :2], w=frame_width, h=frame_height)
        with torch.no_grad():
          model_pos.eval()
          # print (kps_seq_norm.shape)
          # assert False

          batch_2d = np.expand_dims(np.pad(kps_seq_norm[..., :2],
                      ((pad + causal_shift, pad - causal_shift), 
                      (0, 0), (0, 0)), 'edge'), axis=0)

          batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
          batch_2d[1, :, :, 0] *= -1
          batch_2d[1, :, kps_left + kps_right] = batch_2d[1, :, kps_right + kps_left]

          inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
          if torch.cuda.is_available():
              inputs_2d = inputs_2d.to(device)
          inputs_2d = inputs_2d.contiguous()
          # print (inputs_2d.shape)
          # assert False
          predicted_3d_pos = model_pos(inputs_2d)

          # Undo flipping and take average with non-flipped version
          predicted_3d_pos[1, :, :, 0] *= -1
          predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
          predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
          # print ('predicted_3d_pos:', predicted_3d_pos.size())
          prediction = predicted_3d_pos.squeeze(0).cpu().numpy()  
        
        if viz_pose3d:
          axlim = np.amin(prediction), np.amax(prediction)
          parents = skeleton.parents()
          for i, dt in enumerate(dts):
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/pose3d'
            os.makedirs(dir_copy_pid, exist_ok=True)

            pos = prediction[i]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.view_init(elev=125, azim=25)
            ax.set_xlim(axlim)
            ax.set_ylim(axlim)
            ax.set_zlim(axlim)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            for j, j_parent in enumerate(parents):
              if j_parent == -1:
                  continue
              # print (j, j_parent)
                  
              col = 'red' if j in skeleton.joints_right() else 'black'
              ax.plot([pos[j, 0], pos[j_parent, 0]],
                      [pos[j, 1], pos[j_parent, 1]],
                      [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

            plt.tight_layout()
            # plt.show()
            file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            file_save = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
            plt.close('all')
            print ('save in ...', file_save)

        ''' b. PnP 2D & 3d '''
        calib_3d = {'rot': [], 'trans': []}
        for t in range(kps_seq.shape[0]):
          pose2d = kps_seq[t]
          pose3d = prediction[t]
          pose2D_pair = pose2d[idx_2d]
          pose3D_pair = pose3d[idx_3d]

          success, rotation_vector, translation_vector, reprojectionError = cv2.solvePnPGeneric(
                            objectPoints=pose3D_pair,
                            imagePoints=pose2D_pair, 
                            cameraMatrix=camera_matrix, 
                            distCoeffs=dist_coeffs, 
                            useExtrinsicGuess=False,
                            flags=solvepnp_flag)
          reprojectionError = reprojectionError.reshape((-1,))
          # print (len(rotation_vector))
          # print (rotation_vector[0])
          # print (len(translation_vector))
          # print (translation_vector[0])
          # print (reprojectionError.shape)
          # assert False
          sel = np.argmin(reprojectionError)
          rotation_vector = rotation_vector[sel]
          translation_vector = translation_vector[sel]

          calib_3d['rot'].append(rotation_vector)
          calib_3d['trans'].append(translation_vector)
        
        calib_3d['rot'] = np.array(calib_3d['rot'])
        calib_3d['trans'] = np.array(calib_3d['trans'])

        if viz_calib_proj:
          for t in range(kps_seq.shape[0]):
            dt = dts[t]
            pose2d = kps_seq[t]
            pose3d = prediction[t]
            rotation_vector = calib_3d['rot'][t]
            translation_vector = calib_3d['trans'][t]

            imagePoints, jacobian = cv2.projectPoints(
                                      objectPoints=pose3d, 
                                      rvec=rotation_vector, 
                                      tvec=translation_vector, 
                                      cameraMatrix=camera_matrix, 
                                      distCoeffs=dist_coeffs)
            imagePoints = imagePoints.reshape((imagePoints.shape[0], 2))
            # print (pose2d.shape)
            # print (pose3d.shape)
            # print (rotation_vector.shape)
            # print (translation_vector.shape)
            # print (imagePoints.shape)
            # assert False

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
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/calib_proj'
            os.makedirs(dir_copy_pid, exist_ok=True)

            fig, ax = plt.subplots()
            ax.imshow(frame)

            kp = pose2d
            color = 'r'

            for j1, j2 in connection:
              # print (j1, j2)
              # print (kp[j1-1], kp[j2-1])

              y1, x1 = kp[j1-1]
              y2, x2 = kp[j2-1]

              ax.plot(x1, y1, 'o', color=color, markersize=2)
              ax.plot(x2, y2, 'o', color=color, markersize=2)
              ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)

            kp = imagePoints
            color = 'b'

            parents = skeleton.parents()
            for j, j_parent in enumerate(parents):
              if j_parent == -1:
                  continue
              # print (j, j_parent)
              y1, x1 = kp[j]
              y2, x2 = kp[j_parent]

              ax.plot(x1, y1, 'o', color=color, markersize=2)
              ax.plot(x2, y2, 'o', color=color, markersize=2)
              ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
            
            file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_copy_kps_fp_rm_pi)

        if viz_extrinsic:
          dt = dts[0]
          year = dt.year
          month = dt.month
          day = dt.day
          hour = dt.hour
          minute = dt.minute
          second = dt.second
          dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
          if use_ep6_grid:
            dir_copy += '_grid'
          dir_copy += '_ori'
          dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}'
          os.makedirs(dir_copy_pid, exist_ok=True)

          fig, axes = plt.subplots(nrows=2, ncols=1)
          # rotation
          proj_rot = calib_3d['rot']
          xs = np.arange(proj_rot.shape[0])
          axes[0].plot(xs, proj_rot[:,0], c='r', label='rot_x')
          axes[0].plot(xs, proj_rot[:,1], c='g', label='rot_y')
          axes[0].plot(xs, proj_rot[:,2], c='b', label='rot_z')
          axes[0].legend(loc='upper right')
          # translation
          proj_trans = calib_3d['trans']
          xs = np.arange(proj_trans.shape[0])
          axes[1].plot(xs, proj_trans[:,0], c='r', label='trans_x')
          axes[1].plot(xs, proj_trans[:,1], c='g', label='trans_y')
          axes[1].plot(xs, proj_trans[:,2], c='b', label='trans_z')
          axes[1].legend(loc='upper right')

          file_copy_kps_fp_rm_pi = dir_copy_pid + f'/extrinsics.jpg'
          plt.tight_layout()
          plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
          plt.close()
          print ('save in ...', file_copy_kps_fp_rm_pi)
        
        ''' c. smooth calibrated 1) rotation & 2) translation ''' 
        # 1) smooth rotation
        proj_rot = calib_3d['rot']
        proj_rot = proj_rot.reshape((proj_rot.shape[0],3))
        rot = R.from_rotvec(proj_rot)
        quat = rot.as_quat()
        for ch in range(quat.shape[1]):
          quat_ch = quat[:,ch]
          quat_ch = Series(quat_ch).rolling(smooth_calb_len, min_periods=1, center=True).mean().to_numpy()
          quat[:,ch] = quat_ch
        rot = R.from_quat(quat)
        proj_rot = rot.as_rotvec()
        
        calib_3d['rot'] = proj_rot.reshape((proj_rot.shape[0],3,1))
        
        # smoothe trans
        proj_trans = calib_3d['trans']
        proj_trans = proj_trans.reshape((proj_trans.shape[0],3))
        for ch in range(proj_trans.shape[1]):
          trans_ch = proj_trans[:,ch]
          trans_ch = Series(trans_ch).rolling(smooth_calb_len, min_periods=1, center=True).mean().to_numpy()
          proj_trans[:,ch] = trans_ch
        
        calib_3d['trans'] = proj_trans.reshape((proj_trans.shape[0],3,1))

        if viz_extrinsic_smth:
          dt = dts[0]
          year = dt.year
          month = dt.month
          day = dt.day
          hour = dt.hour
          minute = dt.minute
          second = dt.second
          dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
          if use_ep6_grid:
            dir_copy += '_grid'
          dir_copy += '_ori'
          dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}'
          os.makedirs(dir_copy_pid, exist_ok=True)

          fig, axes = plt.subplots(nrows=2, ncols=1)
          # rotation
          proj_rot = calib_3d['rot']
          xs = np.arange(proj_rot.shape[0])
          axes[0].plot(xs, proj_rot[:,0], c='r', label='rot_x')
          axes[0].plot(xs, proj_rot[:,1], c='g', label='rot_y')
          axes[0].plot(xs, proj_rot[:,2], c='b', label='rot_z')
          axes[0].legend(loc='upper right')
          # translation
          proj_trans = calib_3d['trans']
          xs = np.arange(proj_trans.shape[0])
          axes[1].plot(xs, proj_trans[:,0], c='r', label='trans_x')
          axes[1].plot(xs, proj_trans[:,1], c='g', label='trans_y')
          axes[1].plot(xs, proj_trans[:,2], c='b', label='trans_z')
          axes[1].legend(loc='upper right')

          file_copy_kps_fp_rm_pi = dir_copy_pid + f'/extrinsics_smth.jpg'
          plt.tight_layout()
          plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
          plt.close()
          print ('save in ...', file_copy_kps_fp_rm_pi)

        ''' d. calibrate 3d pose 
        The calibrated 3D pose represents the 3D pose view from camera.
        So, the chest vector should be okay to reflect where the person is looking at.
        The noise will be smoothed temporally.

        '''
        
        pose3d_calib = []
        for t in range(prediction.shape[0]):
          pose3d = prediction[t]
          rotation_vector = calib_3d['rot'][t]
          translation_vector = calib_3d['trans'][t]

          cMo = np.eye(4,4)
          Rot = cv2.Rodrigues(rotation_vector)[0]
          cMo[0:3,0:3] = Rot
          cMo[0:3,3] = translation_vector.reshape((3,))

          Rot = cMo[:3,:3]
          if 0:
            Rot = Rot.T
          Trans = cMo[:3,3].reshape((3,1))
          
          if 0: # for orientation translation is not really needed.
            pose3d = np.dot(Rot, pose3d.T)
          else:
            pose3d = np.dot(Rot, pose3d.T) + Trans
          pose3d = pose3d.T
          # pose3d = -pose3d # Need to flip!!!

          pose3d_calib.append(pose3d)

        pose3d_calib = np.array(pose3d_calib)

        if viz_pose3d_calib_match_2d:
          axlim = [np.amin(pose3d_calib), np.amax(pose3d_calib)]
          parents = skeleton.parents()
          for i, dt in enumerate(dts):
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/pose3d_calib'
            os.makedirs(dir_copy_pid, exist_ok=True)

            pos = pose3d_calib[i]
            pose2d = kps_seq[i]
            # print (pose2d)
            # assert False

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.view_init(elev=-90, azim=90)

            for p1, p2 in connection:
              col = 'red' if p1 in kps_right_2d else 'black'
              p1 -= 1
              p2 -= 1
              
              ax.plot(
                [pose2d[p1, 0], pose2d[p2, 0]],
                [pose2d[p1, 1], pose2d[p2, 1]],
                [0, 0],
                color=col)

            if axlim[0] > np.amin(pose2d):
              axlim[0] = np.amin(pose2d)
            if axlim[1] < np.amax(pose2d):
              axlim[1] = np.amax(pose2d)
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                    
                col = 'red' if j in skeleton.joints_right() else 'black'
                ax.plot([pose3d[j, 0], pose3d[j_parent, 0]],
                        [pose3d[j, 1], pose3d[j_parent, 1]],
                        [pose3d[j, 2], pose3d[j_parent, 2]], zdir='z', c=col)

            if axlim[0] > np.amin(pose3d):
              axlim[0] = np.amin(pose3d)
            if axlim[1] < np.amax(pose3d):
              axlim[1] = np.amax(pose3d)

            _idx_2d = np.array([12,14,16,11,13,15,5,7,9,6,8,10])
            _idx_3d = np.array([1,2,3,4,5,6,11,12,13,14,15,16])

            for i in range(len(_idx_2d)):
              # if i != 12:
              #   continue
              i_2d = _idx_2d[i]
              i_3d = _idx_3d[i]
              
              kp_2d = pose2d[i_2d]
              kp_3d = pose3d[i_3d]
              # print (i_2d, i_3d)
              # print (kp_2d, kp_3d)
              
              ax.plot(
                [kp_2d[0], kp_3d[0]],
                [kp_2d[1], kp_3d[1]],
                [0,        kp_3d[2]],
                color='b'
              )

            plt.tight_layout()

            file_save = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_save)
          
        ''' e. Get 3D chest normal vector '''
        left_sh = pose3d_calib[:, 11]
        right_sh = pose3d_calib[:, 14]
        spine = pose3d_calib[:,7]
        thorax = pose3d_calib[:,8]
        left_hip = pose3d_calib[:, 4]
        # print (left_sh.shape)
        # print (right_sh.shape)
        # print (left_hip.shape)
        hori_axis = right_sh - left_sh
        if 1:
          vert_axis = spine - thorax
          chest_start = np.sum(pose3d_calib[:,[11,14,7,8]], axis=1)/4
          # print (pose3d_calib[:,idx_3d].shape)
          # print (chest_start.shape)
          chest_start = chest_start.reshape((-1,1,3))
          # print (chest_start.shape)
        else:
          assert False, 'left and right hip is not on the same plane as there is one more joint (spine) in the middle'
          vert_axis = left_hip - left_sh
          chest_start = np.sum(pose3d_calib[:,idx_3d], axis=1)/4
          # print (pose3d_calib[:,idx_3d].shape)
          # print (chest_start.shape)
          chest_start = chest_start.reshape((-1,1,3))
          # print (chest_start.shape)

        chest_vec = np.cross(hori_axis, vert_axis)
        # print (chest_vec.shape)
        chest_vec = chest_vec.reshape((-1,1,3))
        # print (chest_vec.shape)
        chest_end = chest_vec + chest_start
        # print (chest_end.shape)
        chest_norm_3d = np.concatenate((chest_start, chest_end), axis=1)
        # print (chest_norm_3d.shape)
        # assert False
        
        if 0:
          # Get Rotated chest vector
          chest_end_rot = []
          rot_degree = 90
          rot_rad = np.radians(rot_degree)
          for t in range(chest_norm_3d.shape[0]):
            chest_start_ = chest_start[t].reshape((3,))
            chest_vec_ = chest_vec[t].reshape((3,))
            right_sh_ = right_sh[t]
            left_sh_ = left_sh[t]

            rot_axis_ = right_sh_ - left_sh_
            rot_axis_ /= np.sqrt(np.sum(rot_axis_**2))
            rot_vec = rot_rad * rot_axis_
            rotation = R.from_rotvec(rot_vec)
            chest_vec_rot_ = rotation.apply(chest_vec_)
            chest_end_rot_ = chest_vec_rot_ + chest_start_
            chest_end_rot.append(chest_end_rot_)
          chest_end_rot = np.array(chest_end_rot).reshape((-1,1,3))
          chest_norm_3d_rot = np.concatenate((chest_start, chest_end_rot), axis=1)
        
        if viz_chest_3d:
          axlim = np.amin(pose3d_calib), np.amax(pose3d_calib)
          parents = skeleton.parents()
          for i, dt in enumerate(dts):
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/chest_3d'
            os.makedirs(dir_copy_pid, exist_ok=True)

            pos = pose3d_calib[i]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlim(axlim)
            ax.set_ylim(axlim)
            ax.set_zlim(axlim)
            ax.view_init(elev=125, azim=25)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            
            # pose 3d
            for j, j_parent in enumerate(parents):
              if j_parent == -1:
                  continue
              # print (j, j_parent)
                  
              # col = 'red' if j in skeleton.joints_right() else 'black'
              col='k'
              ax.plot([pos[j, 0], pos[j_parent, 0]],
                      [pos[j, 1], pos[j_parent, 1]],
                      [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
            
            left_sh = pos[11]
            x1, y1, z1 = left_sh
            right_sh = pos[14]
            x2, y2, z2 = right_sh
            if 1:
              spine = pos[7]
              x3, y3, z3 = spine
              thorax = pos[8]
              x4, y4, z4 = thorax
            else:
              left_hip = pos[4]
              x3, y3, z3 = left_hip
              right_hip = pos[1]
              x4, y4, z4 = right_hip
            ax.scatter(x1, y1, z1, s=5, color='r', marker='o', alpha=0.7, label='ori_start')
            ax.scatter(x2, y2, z2, s=5, color='b', marker='o', alpha=0.7, label='ori_start')
            ax.scatter(x3, y3, z3, s=5, color='g', marker='o', alpha=0.7, label='ori_start')
            ax.scatter(x4, y4, z4, s=5, color='g', marker='o', alpha=0.7, label='ori_start')

            # chest 3d
            chest_start = chest_norm_3d[i,0,:]
            chest_end = chest_norm_3d[i,1,:]
            x1, y1, z1 = chest_start
            x2, y2, z2 = chest_end
            # print (chest_start.shape, chest_end.shape)
            # assert False
            ax.plot([chest_start[0], chest_end[0]],
                    [chest_start[1], chest_end[1]],
                    [chest_start[2], chest_end[2]],
                    zdir='z', c='blue')
            ax.scatter(x1, y1, z1, s=5, color='m', marker='o', alpha=0.7, label='ori_start')
            ax.scatter(x2, y2, z2, s=5, color='c', marker='o', alpha=0.7, label='ori_start')

            if 0:
              # chest 3d rotate for sholder axis
              chest_end_rot = chest_norm_3d_rot[i,1,:]

              x1, y1, z1 = chest_start
              x2, y2, z2 = chest_end_rot
              # print (chest_start.shape, chest_end.shape)
              # assert False
              ax.plot([chest_start[0], chest_end_rot[0]],
                      [chest_start[1], chest_end_rot[1]],
                      [chest_start[2], chest_end_rot[2]],
                      zdir='z', c='red')
              ax.scatter(x1, y1, z1, s=5, color='m', marker='o', alpha=0.7, label='ori_start')
              ax.scatter(x2, y2, z2, s=5, color='y', marker='o', alpha=0.7, label='ori_end')
            
            plt.tight_layout()
            file_save = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_save)

        ''' f. Project 3D chest normal vector to x-y plane '''
        chest_3d_vec = chest_norm_3d[:,1] - chest_norm_3d[:,0]
        # print (chest_3d_vec.shape)
        chest_2d_vec = chest_3d_vec[:,:2] 
        # chest_2d_vec *= -1
        # print (chest_2d_vec.shape)
        # chest_2d_vec[:,0] *= -1 # left-right flipped
        chest_2d_norm = np.sqrt(np.sum(chest_2d_vec**2, axis=1)).reshape((-1,1))
        # print (chest_2d_norm.shape)
        chest_2d_vec /= chest_2d_norm
        chest_2d_vec *= 15
        # print (chest_2d_vec.shape)
        # assert False
              
        if viz_chest_2d:
          for i, dt in enumerate(dts):
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/chest_2d'
            os.makedirs(dir_copy_pid, exist_ok=True)

            fig, ax = plt.subplots()
            chest_2d = chest_2d_vec[i]
            x, y = chest_2d
            ax.plot([0, x], [0, y], 'b')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])

            file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.tight_layout()
            plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_copy_kps_fp_rm_pi)

        ''' g. Reflect Camera direction in EP6'''
        if 1:
          degree = camera_degree[pi_ip]
        else:
          degree = 360 - camera_degree[pi_ip]
        theta = np.radians(degree)
        c, s = np.cos(theta), np.sin(theta)
        cam_rot = np.array([
          [c, -s],
          [s, c]])
        # ori_cam = cam_rot @ chest_2d_vec.T 
        # ori_cam = ori_cam.T
        ori_cam = chest_2d_vec @ cam_rot

        if viz_chest_2d_cam:
          for i, dt in enumerate(dts):
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/chest_2d_cam'
            os.makedirs(dir_copy_pid, exist_ok=True)

            fig, ax = plt.subplots()
            chest_2d = ep6_ori[i]
            x, y = chest_2d
            ax.plot([0, x], [0, y], 'b')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])

            file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.tight_layout()
            plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_copy_kps_fp_rm_pi)
        
        ''' h. Project camera-reflected orientation to EP6 axis '''
        x1, y1 = ori_cam[:,0], -ori_cam[:,1] # vertical exis is fliipped in image axis
        x1 = x1.reshape((-1,1))
        y1 = y1.reshape((-1,1))
        ori_ep6 = np.concatenate([x1, y1], axis=1)

        if viz_chest_2d_ep6:
          for i, dt in enumerate(dts):
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/chest_2d_ep6'
            os.makedirs(dir_copy_pid, exist_ok=True)

            fig, ax = plt.subplots()
            chest_2d = ori_ep6[i]
            x, y = chest_2d
            ax.plot([0, x], [0, y], 'b')
            ax.set_xlim([-1, 1])
            ax.set_ylim([1, -1])

            file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            plt.tight_layout()
            plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_copy_kps_fp_rm_pi)

        ''' i. Project Foot to EP6 map '''
        avg_feet_positions = extract_avg_feet_positions(kps_seq)
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

        if viz_occup_ori_ep6_pi:
          for t in range(len(ori_ep6)):
            dt = dts[t]
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
            if use_ep6_grid:
              dir_copy += '_grid'
            dir_copy += '_ori'
            dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/occup_ori_ep6'
            os.makedirs(dir_copy_pid, exist_ok=True)

            ori = ori_ep6[t]
            occup = EP6_feet_pos[t]
            x1, y1 = ori
            x2, y2 = occup

            # ori_end = ori + occup
            x3, y3 = x2 + x1, y2 + y1
            # # dir_copy_pid += '_occup+ori'
            
            # # ori_end = occup - ori
            # x3, y3 = x2 - x1, y2 - y1
            # dir_copy_pid += '_occup-ori'

            # # ori_end = occup - ori_swap
            # x3, y3 = x2 - y1, y2 - x1
            # dir_copy_pid += '_occup-ori_swap'

            # # ori_end = occup - ori-x+y
            # x3, y3 = x2 - x1, y2 + y1
            # dir_copy_pid += '_occup-ori-x+y'

            # os.makedirs(dir_copy_pid, exist_ok=True)

            fig, ax = plt.subplots()
            ax.imshow(ep6_map)
            ax.scatter(x2, y2, s=5, color='r', marker='o', alpha=0.7, label='occup')

            ax.plot([x2, x3], [y2, y3], color='b')

            # ax.legend(loc='upper right')

            file_copy_occup_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
            # plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
            plt.close()
            print ('save in ...', file_copy_occup_pi)


        ''' collect 1) ori & 2) foot '''
        for i, dt in enumerate(dts):
          # 1) ori
          if dt not in ori_pi:
            ori_pi[dt] = {}
          if pi not in ori_pi[dt]:
            ori_pi[dt][pi] = np.empty((0,2))
          ori = ori_ep6[i].reshape((1,2))
          ori_pi[dt][pi] = np.concatenate((ori_pi[dt][pi], ori))

          # 2) foot
          if dt not in occup_pi:
            occup_pi[dt] = {}
          if pi not in occup_pi[dt]:
            occup_pi[dt][pi] = np.empty((0,2))
          occup = EP6_feet_pos[i].reshape((1,2))
          occup_pi[dt][pi] = np.concatenate((occup_pi[dt][pi], occup))

if viz_occup_ori:
  ''' draw occupancy & Orientation observed from all Pis '''
  list_pi = list(pi_data.keys())
  list_pi.sort()
  c_pis = cmap(np.linspace(0, 1, len(list_pi)))
  pi_color = {}
  for pi, c in zip(list_pi, c_pis):
    pi_color[pi] = c

  for dt in ori_pi:
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_occup = dir_copy + '/occup'

    # per pi
    for pi in ori_pi[dt]:
      dir_copy_occup_pi = dir_copy_occup + f'/{pi}'
      os.makedirs(dir_copy_occup_pi, exist_ok=True)  

      occup = occup_pi[dt][pi]
      ori = ori_pi[dt][pi]

      fig, ax = plt.subplots()
      ax.imshow(ep6_map)
      for i in range(occup.shape[0]):
        color = c_kps[i]
        occup_ = occup[i]
        ori_ = ori[i]
        ori_start = ori_[0]
        ori_end = ori_[1]

        x1, y1 = ori_start
        x2, y2 = ori_end
        x3, y3 = occup_

        ax.scatter(x1, y1, s=5, color='r', marker='o', alpha=0.7, label='ori_start')
        ax.scatter(x2, y2, s=5, color='g', marker='X', alpha=0.7, label='ori_end')
        ax.scatter(x3, y3, s=5, color='b', marker='^', alpha=0.7, label='occup')

        ax.plot([x1, x2], [y1, y2], color='m')
        ax.plot([x2, x3], [y2, y3], color='c')

      file_copy_occup_pi = dir_copy_occup_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
      plt.close()
      print ('save in ...', file_copy_occup_pi)

    # all pis
    dir_copy_occup_indiv = dir_copy_occup + '/individual'
    os.makedirs(dir_copy_occup_indiv, exist_ok=True)

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for pi in occup_pi[dt]:
          
      occup = occup_pi[dt][pi]
      ori = ori_pi[dt][pi]
      ori_start = ori[:,0,:]
      ori_end = ori[:,1,:]

      color = pi_color[pi]

      ax.scatter(ori_start[:,0], ori_start[:,1], s=10, color=color, marker='o', alpha=0.7)
      ax.scatter(ori_end[:,0], ori_end[:,1], s=10, color=color, marker='X', alpha=0.7)
      ax.scatter(occup[:,0], occup[:,1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

    ax.legend()

    file_copy_occup_pi = dir_copy_occup_indiv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  assert False

if via_kps_track | viz_pose3d | viz_calib_proj | viz_extrinsic \
  | viz_extrinsic_smth | viz_pose3d_calib \
  | viz_pose3d_calib_match_2d | viz_chest_3d \
  | viz_chest_2d | viz_chest_2d_cam | viz_chest_2d_ep6 \
  | viz_occup_ori_ep6_pi:
  assert False

''' 
Multi-view: Person matching and association across camera 

Current version:

Multi-view Localization:
- Only use occupation
TODO: will update if the orientation looks okay.
dist = L_2(occup_i, occup_j) + lambda * -cosine_similarity

Multi-view Orientation:
Since multi-view localization does not use orientation yet,
Just get the average vector of connected components.
TODO: Once multi-view localization is updated with orientation, then only take average of those are connected for this case.

'''

list_pi = list(pi_data.keys())
list_pi.sort()

occup_mv = {}
ori_mv = {}
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
  ori_mv[dt] = []
  for cc in nx.connected_components(G):
    cc = list(cc)
    # print (cc)

    n_node = 0
    occup = np.zeros((2,))
    ori = np.zeros((2,))
    for pi_o in cc:
      pi, i = cc[0].split('_')
      i = int(i)
      _occup = occup_pi[dt][pi][i]
      # print (occup.shape)
      # assert False
      occup += _occup
      n_node += 1

      # get orientation vectors as average of connected components
      _ori = ori_pi[dt][pi][i]
      _ori_vec = _ori[1] - _ori[0]
      ori += _ori_vec

    occup /= n_node
    occup_mv[dt].append(occup)

    ori /= n_node
    ori /= np.sqrt(np.sum(ori**2))

    if 0:
      # separate start & end points
      _ori = np.zeros((2,2))
      _ori[0,0] = occup # ori start
      _ori[0,1] = occup + ori

      ori_mv[dt].append(_ori)
    else:
      # only the vector
      ori_mv[dt].append(ori)
    assert False, 'Need to figure this out.'

  occup_mv[dt] = np.array(occup_mv[dt])
  ori_mv[dt] = np.array(ori_mv[dt])

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
    dir_copy += '_ori'
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

    ori_pi_dt = ori_pi[dt]

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    for node in G.nodes(data=True):
      pi, i = node[0].split('_')
      pi_ip = pi[2:5]
      loc = node[1]['loc']
      color = pi_color[pi]
      ax.scatter(loc[0], loc[1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

      # draw orientation
      _ori = ori_pi_dt[pi][i]
      x1, y1 = ori
      x2, y2 = loc
      # ori_end = ori + occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='b')
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
  ''' draw occupancy & orientation with multi-view '''
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  dir_copy += '_ori'
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
    ori = ori_mv[dt]
    
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

    # orientation
    for i in range(ori.shape[0]):
      ori_i = ori[i]
      occup_i = occup[i]
      x1, y1 = ori_i
      x2, y2 = occup_i
      
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='b')      

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
dir_copy += '_ori'
dir_save = dir_copy + '/occup'

os.makedirs(dir_save, exist_ok=True)

file_save = dir_save + '/multi-view.p'
cp.dump(occup_mv, open(file_save, 'wb'))
print ('save in ...', file_save)

file_save = dir_save + '/multi-view_ori.p'
cp.dump(ori_mv, open(file_save, 'wb'))
print ('save in ...', file_save)

#-------------------------------------------------

''' Post-Process Multi-view: Temporal Smoothing
1) Remove False Negatives: 2nd order interpolation
2) Remove False Negatives: 1st order interpolation

Current version,
Only considers the location.
TODO: Need to consider the orientation for selecting the interpolated samples.
'''

''' 1) Remove False Negatives: 2nd order interpolation  '''
list_dt = list(occup_mv.keys())
list_dt.sort()
for t in range(1, len(list_dt)-1):
  dt_curr = list_dt[t]
  dt_prev = list_dt[t-1]
  dt_next = list_dt[t+1]

  w = (dt_curr - dt_prev).seconds/(dt_next - dt_prev).seconds

  ori_curr = ori_mv[dt_curr]
  ori_prev = ori_mv[dt_prev]
  ori_next = ori_mv[dt_next]

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

      # orientation
      or_prev = ori_prev[r]
      or_next = or_next[c]
      or_curr = (1-w)*or_prev + w*or_next
      or_curr /= np.sqrt(np.sum(or_curr**2))
      or_curr = or_curr.reshape((1, 2))
      or_curr = np.concatenate((ori_curr, or_curr))

  occup_mv[dt_curr] = occup_curr  
  ori_mv[dt_curr] = or_curr

# remove empty frame
occup_mv_clean = {}
ori_mv_clean = {}
for dt in occup_mv:
  occup = occup_mv[dt]
  ori = ori_mv[dt]
  if occup.shape[0] > 0:
    occup_mv_clean[dt] = occup
    ori_mv_clean[dt] = ori
occup_mv = occup_mv_clean
ori_mv = ori_mv_clean

if viz_occup_fn1:
  ''' draw occupancy with multi-view '''
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  dir_copy += '_ori'
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

    # orientation
    occup = occup_mv[dt]
    ori = ori_mv[dt]
    x1, y1 = ori
    x2, y2 = occup
    x3, y3 = x2 + x1, y2 + y1

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

    ax.plot([x2, x3], [y2, y3], color='b')      

    file_copy_occup_pi = dir_copy_occup_mv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  
  assert False

#-------------------------------------------------


''' 2) Remove False Negatives: 1st order interpolation 

Current version:
Select interpolating samples between frames based on locations

TODO: also include orientation similarity in distance metric
'''
list_dt = list(occup_mv.keys())
list_dt.sort()
for t in range(len(list_dt)-1):
  dt_prev = list_dt[i]
  dt_next = list_dt[i+1]
  interval = (dt_next - dt_prev).seconds
  if interval == 1:
    continue

  ori_prev = ori_mv[dt_prev]
  ori_next = ori_mv[dt_next]

  occup_prev = occup_mv[dt_prev]
  occup_next = occup_mv[dt_next]
  row_int, col_ind = occup_matching_between_frames(occup_prev, occup_next, th=overlap_th)

  list_dt_interp = list(timerange(dt_prev, dt_next)) # dt_next not included
  for t in range(1, len(list_dt_interp)):
    w = t/(len(list_dt_interp))
    dt_curr = list_dt_interp[t]

    ori_curr = np.empty((0,2))

    occup_curr = np.empty((0,2))
    for r,c in zip(row_int, col_ind):
      oc_prev = occup_prev[r]
      oc_next = occup_next[c]
      oc_curr = (1-w)*oc_prev + w*oc_next
      oc_curr = oc_curr.reshape((1,2))
      occup_curr = np.concatenate(occup_curr, oc_curr)

      # orientation
      or_prev = ori_prev[r]
      or_next = ori_next[c]
      or_curr = (1-w)*or_prev + w*or_next
      or_curr /= np.sqrt(np.sum(or_curr**2))
      or_curr = or_curr.reshape((1, 2))
      or_curr = np.concatenate((ori_curr, or_curr))
    
    if occup_curr.shape[0] > 0:
      occup_mv[dt_curr] = occup_curr
      ori_mv[dt_curr] = or_curr

# remove empty frame
occup_mv_clean = {}
ori_mv_clean = {}
for dt in occup_mv:
  occup = occup_mv[dt]
  ori = ori_mv[dt]
  if occup.shape[0] > 0:
    occup_mv_clean[dt] = occup
    ori_mv_clean[dt] = ori
occup_mv = occup_mv_clean
ori_mv = ori_mv_clean

if viz_occup_fn2:
  ''' draw occupancy with multi-view '''
  dir_copy = dir_exp + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  if use_ep6_grid:
    dir_copy += '_grid'
  dir_copy += '_ori'
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
    ori = ori_mv[dt]
    x1, y1 = ori
    x2, y2 = occup
    x3, y3 = x2 + x1, y2 + y1

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

    ax.plot([x2, x3], [y2, y3], color='b')      

    file_copy_occup_pi = dir_copy_occup_mv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
  
  assert False

#--------------------------------------------

''' 
Tracking with Hungarian Method 

Current version:
Only track with locations
TODO: need to include orientation in the distance metric
'''
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
    ori_curr = ori_mv[dt_curr]
    pid = tracker_occup.update(occup_curr)
    for rid, pid  in enumerate(pid):
      if pid not in mv_seq:
        mv_seq[pid] = {
          'dt': [],
          'rid': [],
          'loc': [],
          'ori': []}
      if max_pid is None:
        max_pid = pid
      elif max_pid < pid:
        max_pid = pid
      # print (max_pid, pid)

      mv_seq[pid]['dt'].append(dt_curr)
      mv_seq[pid]['rid'].append(rid)
      mv_seq[pid]['loc'].append(occup_curr[rid])
      mv_seq[pid]['ori'].append(ori_curr[rid])

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}'
  
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)
    
    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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
    for dt_curr, rid, loc, ori in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc'],
                                mv_seq[pid]['ori']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': [],
                              'ori': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)
      mv_seq_dt[dt_curr]['ori'].append(ori)

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
    dir_copy += '_ori'
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}'
    
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr, ori_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc'],
                            mv_seq_dt[dt_curr]['ori']
                            ):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      ori_prev = mv_seq_dt[dt_prev]['ori'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]

      ax.plot(xs, ys, c='k')
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)

      # orientation
      assert False
    
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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_fp'
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_fp'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])    

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

    file_save = dir_save + f'/id_{int(pid):03d}.png'
    plt.savefig(file_save)
    plt.close('all')  
    print ('save in ...', file_save)
  # assert False
  
  ''' draw for each second for resulting tracks '''
  # each second
  mv_seq_dt = {}
  for pid in mv_seq:
    for dt_curr, rid, loc, ori in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc'],
                                mv_seq[pid]['ori']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': [],
                              'ori': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)
      mv_seq_dt[dt_curr]['ori'].append(ori)

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
    dir_copy += '_ori'
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_fp'
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr, ori_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc'],
                            mv_seq_dt[dt_curr]['ori']):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      ori_prev = mv_seq_dt[dt_prev]['ori'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]
      ax.plot(xs, ys, c='k', label=pid)
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)

      # orientation
      assert False
    
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

  # smoothe location
  locs = np.array(mv_seq[pid]['loc'])
  for ch in range(locs.shape[1]):
    locs_ch = locs[:, ch]
    locs_ch = Series(locs_ch).rolling(smooth_track_len, min_periods=1, center=True).mean().to_numpy()
    locs[:, ch] = locs_ch
  mv_seq[pid]['loc'] = locs

  # smooth orientation
  oris = np.array(mv_seq[pid]['ori'])
  for ch in range(oris.shape[1]):
    locs_ch = oris[:, ch]
    locs_ch = Series(locs_ch).rolling(smooth_track_len, min_periods=1, center=True).mean().to_numpy()
    oris[:, ch] = locs_ch
  mv_seq[pid]['ori'] = oris

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_smth'
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_smth'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

    file_save = dir_save + f'/id_{int(pid):03d}.png'
    plt.savefig(file_save)
    plt.close('all')  
    print ('save in ...', file_save)
  # assert False
  
  ''' draw for each second for resulting tracks '''
  # each second
  mv_seq_dt = {}
  for pid in mv_seq:
    for dt_curr, rid, loc, ori in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc'],
                                mv_seq[pid]['ori']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': [],
                              'ori': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)
      mv_seq_dt[dt_curr]['ori'].append(ori)

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
    dir_copy += '_ori'
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_smth'
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr, ori_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc'],
                            mv_seq_dt[dt_curr]['ori']):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      ori_prev = mv_seq_dt[dt_prev]['ori'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]
      ax.plot(xs, ys, c=color, label=pid)
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)

      # orientation
      assert False      
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)

#--------------------------------------------
'''
Merge Frame-based orientation with tracking-based orientation

1) For each frame get tracking-directions
2) weighted average of tracking-based & frame-based orientation
'''
for pid in mv_seq:
  locs = np.array(mv_seq[pid]['loc']) # T x 2

  oris_track = np.empty(mv_seq[pid]['ori'].shape) # T x 2
  oris_track[:] = np.nan
  for j in range(len(locs)-1):
    loc_curr = locs[j]
    loc_next = locs[j+1]
    ori_curr = loc_next - loc_curr
    ori_curr /= np.sqrt(np.sum(ori_curr*2))
    
    oris_track[j] = ori_curr
  oris_track[j+1] = oris_track[j]
  assert np.all(np.isfinite(oris_track))

  oris_frame = np.array(mv_seq[pid]['ori'])
  mv_seq[pid]['ori'] = w_tr*oris_track + oris_frame

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_ori_track_w_{w_tr}'
  
  dir_save = dir_copy_track
  os.makedirs(dir_save, exist_ok=True)

  fig, ax = plt.subplots()
  ax.imshow(ep6_map)
  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)
    
    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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
  dir_copy += '_ori'
  dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_ori_track_w_{w_tr}'
  
  dir_save = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_save, exist_ok=True)

  for pid in mv_seq:
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc'])
    oris = np.array(mv_seq[pid]['ori'])

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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
    for dt_curr, rid, loc, ori in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['rid'],
                                mv_seq[pid]['loc'],
                                mv_seq[pid]['ori']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'rid': [],
                              'loc': [],
                              'ori': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['rid'].append(rid)
      mv_seq_dt[dt_curr]['loc'].append(loc)
      mv_seq_dt[dt_curr]['ori'].append(ori)

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
    dir_copy += '_ori'
    dir_copy_track = dir_copy + f'/track_th_{tracklet_th}_ori_track_w_{w_tr}'
    
    dir_save = dir_copy_track + '/seconds'
    os.makedirs(dir_save, exist_ok=True)

    file_save = dir_save + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      continue

    for pid, loc_curr, ori_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc'],
                            mv_seq_dt[dt_curr]['ori']
                            ):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      ori_prev = mv_seq_dt[dt_prev]['ori'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]

      ax.plot(xs, ys, c='k')
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)

      # orientation
      assert False
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)
  
  # assert False

#--------------------------------------------

'''
Group detection
Graph-based approach: 
Connected components are groups.
Clque is very conservative approach.
'''

# Get per frame occupancy from the cleaned trajectory
occup_track_cleaned = {}
ori_track_cleaned = {}
pid_track_cleaned = {}
for pid in mv_seq:
  dts = mv_seq[pid]['dt']
  locs = mv_seq[pid]['loc']
  oris = mv_seq[pid]['ori']

  for dt, loc, ori in zip(dts, locs, oris):
    if dt not in occup_track_cleaned:
      occup_track_cleaned[dt] = np.empty((0,2))
      ori_track_cleaned[dt] = np.empty((0,2))
      pid_track_cleaned[dt] = []
    
    occup = occup_track_cleaned[dt]
    loc = loc.reshape((1,2))
    occup = np.concatenate((occup, loc))
    occup_track_cleaned[dt] = occup

    orient = ori_track_cleaned[dt]
    ori = ori.reshape((1,2))
    orient = np.concatenate((orient, ori))
    ori_track_cleaned[dt] = orient

    pid_track_cleaned[dt].append(pid)

# Find one-on-one interaction
interactions = {}
for dt in occup_track_cleaned:

  occup = occup_track_cleaned[dt]
  ori = ori_track_cleaned[dt]
  pid = pid_track_cleaned[dt]

  G = nx.Graph()
  for i in range(occup.shape[0]):
    G.add_node(i, loc=occup[i], ori=ori[i], pid=pid[i])
  
  for i in range(occup.shape[0]-1):
    for j in range(i+1, occup.shape[0]):
      loc_i = occup[i]
      loc_j = occup[j]
      dist = np.sqrt(np.sum((loc_i - loc_j)**2))

      ori_i = ori[i]
      ori_j = ori[j]
      facing_inter = detect_direct_interaction(loc_i, ori_i, loc_j, ori_j)

      if (dist < interaction_th) and facing_inter:
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
  dir_copy += '_ori'
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
    orient = ori_track_cleaned[dt]

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7) 

    # orientation
    for j in range(len(occup)):
      occ = locs[j]
      ori = orient[j]
      x1, y1 = ori
      x2, y2 = occ
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

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

#--------------------------------------------