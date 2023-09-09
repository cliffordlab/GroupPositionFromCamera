''' 2) Remove False Negatives: 1st order interpolation 

Current version:
Select interpolating samples between frames based on locations

TODO: also include orientation similarity in distance metric
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from debug import viz_occup_mv_fn

def mv_fn_2(occup_mv, ori_mv, cfg):
  list_dt = list(occup_mv.keys())
  list_dt.sort()
  for t in range(len(list_dt)-1):
    dt_prev = list_dt[t]
    dt_next = list_dt[t+1]
    interval = (dt_next - dt_prev).seconds
    if interval == 1:
      continue

    ori_prev = ori_mv[dt_prev]
    ori_next = ori_mv[dt_next]

    occup_prev = occup_mv[dt_prev]
    occup_next = occup_mv[dt_next]
    row_int, col_ind = occup_matching_between_frames(occup_prev, occup_next, th=overlap_th)

    list_dt_interp = list(timerange(dt_prev, dt_next)) # dt_next not included
    for i in range(1, len(list_dt_interp)):
      w = i/(len(list_dt_interp))
      dt_curr = list_dt_interp[i]

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
        or_curr_norm = np.sqrt(np.sum(or_curr**2))
        if or_curr_norm > 0:
          or_curr /= or_curr_norm
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

  if cfg.viz_occup_mv_fn2:
    viz_occup_mv_fn(occup_mv, ori_mv, f'multi-view_ov_{cfg.overlap_th}_fn2', cfg)
        
  if cfg.stop_after_viz:
    if cfg.viz_occup_mv_fn2:
      assert False
    
  return occup_mv, ori_mv