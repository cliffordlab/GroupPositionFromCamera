''' 2-1) Resolve Temporal False Negative  '''

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

def kps_fn_1(pi_data, cfg):
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
  if cfg.viz_kps_fn1:
    for pi in pi_data:
      for dt_curr in pi_data[pi]:
        kps = pi_data[pi][dt_curr]
        viz_kps(dt_curr, pi, kps, 'kps_fn1', cfg)
    
  if cfg.stop_after_viz:
    if cfg.viz_kps_fn1:
      assert False

  return pi_data