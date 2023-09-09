''' 3) Smooth noisy poses (5 second window) '''

import numpy as np
from datetime import datetime
from itertools import groupby
from operator import itemgetter

import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from pandas import Series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from debug import viz_kps

def kps_smth(pi_data, cfg):
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
            kps_seq_ch = Series(kps_seq_ch).rolling(cfg.smooth_kps_len, min_periods=1, center=True).mean().to_numpy()
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
  if cfg.viz_kps_smth:
    c_kps = cfg.cmap(np.linspace(0, 1, 10))    
    for pi in pi_data:
      for dt in pi_data[pi]:
        kps = pi_data[pi][dt]
        viz_kps(dt, pi, kps, 'kps_smth', cfg)
  
  if cfg.stop_after_viz:
    if cfg.viz_kps_smth:
      assert False

  return pi_data