''' Post-Process tracking: Temporal Smoothing
Temporal Smoothing
'''

import numpy as np
from pandas import Series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from debug import viz_track

def track_smth(mv_seq, cfg):
  # delta_minute = timedelta(minutes=1) # per-minute
  delta_second = timedelta(seconds=1) # per-minute
  
  for pid in mv_seq:
    # smoothe location
    locs = np.array(mv_seq[pid]['loc'])
    for ch in range(locs.shape[1]):
      locs_ch = locs[:, ch]
      locs_ch = Series(locs_ch).rolling(cfg.smooth_track_len, min_periods=1, center=True).mean().to_numpy()
      locs[:, ch] = locs_ch
    mv_seq[pid]['loc'] = locs

    # smooth orientation
    oris = np.array(mv_seq[pid]['ori'])
    for ch in range(oris.shape[1]):
      locs_ch = oris[:, ch]
      locs_ch = Series(locs_ch).rolling(cfg.smooth_track_len, min_periods=1, center=True).mean().to_numpy()
      oris[:, ch] = locs_ch
    mv_seq[pid]['ori'] = oris

  if cfg.viz_tracking_smth:
    viz_track(mv_seq, 'track_fp_smth', cfg)

  if cfg.stop_after_viz:
    if cfg.viz_tracking_smth:
      assert False

  return mv_seq