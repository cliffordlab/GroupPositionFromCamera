''' 1-1) Resolve Temporal False Positive '''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
#from utils.tracking import *
from debug import viz_kps

def kps_fp_1(pi_data, cfg):
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
  if cfg.viz_kps_fp1:    
    for pi in pi_data:
      for dt_curr in pi_data[pi]:
        kps = pi_data[pi][dt_curr]
        viz_kps(dt_curr, pi, kps, 'kps_fp1', cfg)

  if cfg.stop_after_viz:
    if cfg.viz_kps_fp1:
      assert False

  return pi_data