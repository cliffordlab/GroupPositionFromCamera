from utils.tracking import timerange
from tracking.kf.kf_occup_ori import Tracker
import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from debug import viz_track

def mv_smth_kf(occup_mv, ori_mv, cfg):

  list_t = list(occup_mv)
  t_min, t_max = np.amin(list_t), np.amax(list_t)

  tracker = Tracker(cfg.mv_smooth_dist_th, 
                    cfg.mv_smooth_max_frame_skip,
                    collect_states=True,
                    w_ori_tr=cfg.w_tr)

  for dt in timerange(t_min, t_max):
    if dt in occup_mv:
      occup = occup_mv[dt]
      ori = ori_mv[dt]
      # print (ori.shape)
      ori_norm = np.sqrt(np.sum(ori**2, axis=1)). reshape((-1,1))
      ori_norm[ori_norm==0] = 1.
      ori /= ori_norm

      centers = np.concatenate([occup, ori], axis=1)

      tracker.update(centers, dt)
    else:
      tracker.update(ts=dt)

  tracker.close()

  mv_seq = {}
  for trackId in tracker.tracks_collected:
    ts = tracker.tracks_collected[trackId].ts
    center = tracker.tracks_collected[trackId].means_smoothed[:,:4]
    center = np.array(center)

    if center.shape[0] < cfg.min_track_len: 
      # too short to keep track
      continue
    
    if trackId not in mv_seq:
      mv_seq[trackId] = {'dt': [],
                         'loc': [],
                         'ori': []}      

    for i, dt in enumerate(ts):
      mv_seq[trackId]['dt'].append(dt)
      
      occup = center[i,:2].reshape((1,2))
      mv_seq[trackId]['loc'].append(occup)
      
      ori = center[i, 2:].reshape((1,2))
      ori_norm = np.sqrt(np.sum(ori**2))
      if ori_norm == 0:
        ori_norm = 1.
      ori /= ori_norm

      mv_seq[trackId]['ori'].append(ori)

  if cfg.viz_tracking:
    viz_track(mv_seq, 'track_kf', cfg)

  return mv_seq
