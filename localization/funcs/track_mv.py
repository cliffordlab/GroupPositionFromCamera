''' 
Tracking with Hungarian Method 

Current version:
Only track with locations
TODO: need to include orientation in the distance metric
'''
from datetime import datetime, timedelta
import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from itertools import groupby
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from debug import viz_track

def track_mv(occup_mv, ori_mv, cfg):
  # delta_minute = timedelta(minutes=1) # per-minute
  delta_second = timedelta(seconds=1) # per-minute
  
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
                                    th=cfg.tracklet_th)
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

  if 0:
    print ('--------')
    for pid in mv_seq:
      print (pid, len(mv_seq[pid]['loc']))
    print ('--------')
    # assert False

  if cfg.viz_tracking:
    viz_track(mv_seq, 'track_mv', cfg)
    
  if cfg.stop_after_viz:
    if cfg.viz_tracking:
      assert False
  
  return mv_seq