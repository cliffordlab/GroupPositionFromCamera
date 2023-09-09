''' 2-2) Resolve Temporal False Negative '''

import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import gc
from debug import viz_kps

def kps_fn_2(pi_data, cfg):
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
  if cfg.viz_kps_fn2:    
    for pi in pi_data:
      for dt_curr in pi_data[pi]:
        kps = pi_data[pi][dt_curr]
        viz_kps(dt_curr, pi, kps, 'kps_fn2', cfg)
          
  if cfg.stop_after_viz:
    if cfg.viz_kps_fn2:
      assert False

  return pi_data