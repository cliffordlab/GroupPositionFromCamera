'''
Merge Frame-based orientation with tracking-based orientation

1) For each frame get tracking-directions
2) weighted average of tracking-based & frame-based orientation
'''
import numpy as np
import pickle as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from debug import viz_track

def track_ori(mv_seq, cfg):
  # delta_minute = timedelta(minutes=1) # per-minute
  delta_second = timedelta(seconds=1) # per-minute
  
  for pid in mv_seq:
    locs = np.array(mv_seq[pid]['loc']) # T x 2

    oris_track = np.empty(mv_seq[pid]['ori'].shape) # T x 2
    oris_track[:] = np.nan
    for j in range(len(locs)-1):
      loc_curr = locs[j]
      loc_next = locs[j+1]
      ori_curr = loc_next - loc_curr
      try:
        ori_norm = np.sqrt(np.sum(ori_curr**2))
      except:
        print(ori_curr)
        print (ori_curr**2)
        print (np.sum(ori_curr**2))
        print (np.sqrt(np.sum(ori_curr**2)))
        assert False
      # print (ori_curr)
      # print (ori_norm)
      # assert False
      if ori_norm > 0:
        ori_curr /= ori_norm
      
      oris_track[j] = ori_curr
    oris_track[j+1] = oris_track[j]
    assert np.all(np.isfinite(oris_track))

    oris_frame = np.array(mv_seq[pid]['ori'])
    mv_seq[pid]['ori'] = cfg.w_tr*oris_track + oris_frame
  
  if cfg.viz_tracking_ori:
    viz_track(mv_seq, 'track_fp_smth_ori', cfg)
      
  if cfg.stop_after_viz:
    if cfg.viz_tracking_ori:
      assert False
  
  return mv_seq