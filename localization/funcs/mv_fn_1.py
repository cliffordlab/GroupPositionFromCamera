''' 1) Remove False Negatives: 2nd order interpolation  '''

import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from debug import viz_occup_mv_fn

def mv_fn_1(occup_mv, ori_mv, cfg):
  list_dt = list(occup_mv.keys())
  list_dt.sort()
  for t in range(1, len(list_dt)-1):
    dt_curr = list_dt[t]
    dt_prev = list_dt[t-1]
    dt_next = list_dt[t+1]

    w = (dt_curr - dt_prev).seconds/(dt_next - dt_prev).seconds

    ori_curr = ori_mv[dt_curr]
    ori_prev = ori_mv[dt_prev]
    ori_next = ori_mv[dt_next]

    occup_curr = occup_mv[dt_curr]
    occup_prev = occup_mv[dt_prev]
    occup_next = occup_mv[dt_next]

    row_prev, col_curr = occup_matching_between_frames(occup_prev, occup_curr, th=cfg.overlap_th)

    row_curr, col_next = occup_matching_between_frames(occup_curr, occup_next, th=cfg.overlap_th)

    row_pprev, col_nnext = occup_matching_between_frames(occup_prev, occup_next, th=cfg.overlap_th)

    # if matching exist between -/+1 frames, but no matching with 0 frame, then it is false negative
    for r, c in zip(row_pprev, col_nnext):
      if r not in row_prev \
      and c not in col_next:
        oc_prev = occup_prev[r]
        oc_next = occup_next[c]
        oc_curr = (1-w)*oc_prev + w*oc_next
        oc_curr = oc_curr.reshape((1, 2))
        occup_curr = np.concatenate((occup_curr, oc_curr))

        # orientation
        or_prev = ori_prev[r]
        or_next = ori_next[c]
        or_curr = (1-w)*or_prev + w*or_next
        or_norm = np.sqrt(np.sum(or_curr**2))
        if or_norm > 0:
          or_curr /= or_norm
        or_curr = or_curr.reshape((1, 2))
        ori_curr = np.concatenate((ori_curr, or_curr))

    occup_mv[dt_curr] = occup_curr  
    ori_mv[dt_curr] = ori_curr

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

  if cfg.viz_occup_mv_fn1:
    viz_occup_mv_fn(occup_mv, ori_mv, f'multi-view_ov_{cfg.overlap_th}_fn1', cfg)
    
  if cfg.stop_after_viz:
    if cfg.viz_occup_mv_fn1:
      assert False

  return occup_mv, ori_mv