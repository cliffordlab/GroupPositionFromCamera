''' 1-2) Resolve Temporal False Positive '''

import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from debug import viz_kps

def kps_fp_2(pi_data, cfg):
  # delta_minute = timedelta(minutes=1) # per-minute
  delta_second = timedelta(seconds=1) # per-minute
  
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
  if cfg.viz_kps_fp2:
    for pi in pi_data:
      for dt_curr in pi_data[pi]:
        kps = pi_data[pi][dt_curr]
        viz_kps(dt_curr, pi, kps, 'kps_fp2', cfg)
  
  if cfg.stop_after_viz:
    if cfg.viz_kps_fp2:
      assert False

  return pi_data