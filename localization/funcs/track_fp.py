''' Post-Process tracking:
Remove False Positives: Remove short tracklets (<= 5 sec)
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from debug import viz_track

def track_fp(mv_seq, cfg):
  # delta_minute = timedelta(minutes=1) # per-minute
  delta_second = timedelta(seconds=1) # per-minute
  
  mv_seq_cleaned = {}
  for pid in mv_seq:
    if len(mv_seq[pid]['loc']) > cfg.min_track_len:
      mv_seq_cleaned[pid] = mv_seq[pid]
      # print (pid, len(mv_seq[pid]['loc']))
      # print (mv_seq_cleaned[pid]['loc'])
      # print (mv_seq[pid]['loc'])
      # print ('----------------------')
  mv_seq = mv_seq_cleaned
  # assert False

  if 0:
    print ('--------')
    for pid in mv_seq:
      print (pid, len(mv_seq[pid]['loc']))
    print ('--------')
    # assert False

  if cfg.viz_tracking_fp:
    viz_track(mv_seq, 'track_fp', cfg)
      
  if cfg.stop_after_viz:
    if cfg.viz_tracking_fp:
      assert False
    
  return mv_seq