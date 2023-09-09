'''
Smooth location and orientation per Pi
before multi-view integration
'''

from utils.tracking import timerange
from tracking.kf.kf_occup_ori import Tracker
import numpy as np

def occup_ori_smth_kf(occup_pi, ori_pi, ori_pi_2d, pi_data, cfg):

  # rearrange dt->pi to pi->dt
  occup_dt = {}
  for dt in occup_pi:
    for pi in occup_pi[dt]:
      if pi not in occup_dt:
        occup_dt[pi] = {}
      
      occup = occup_pi[dt][pi]
      occup_dt[pi][dt] = occup
  
  ori_dt = {}
  for dt in ori_pi:
    for pi in ori_pi[dt]:
      if pi not in ori_dt:
        ori_dt[pi] = {}
      
      ori = ori_pi[dt][pi]
      ori_dt[pi][dt] = ori
      
  ori_2d_dt = {}
  for dt in ori_pi_2d:
    for pi in ori_pi_2d[dt]:
      if pi not in ori_2d_dt:
        ori_2d_dt[pi] = {}
      
      ori_2d = ori_pi_2d[dt][pi]
      ori_2d_dt[pi][dt] = ori_2d
  
  # kalman filtering & smoothing
  occup_pi = {}
  ori_pi = {}
  #ori_pi_2d = {}
  for pi in occup_dt:
    list_t = list(occup_dt[pi])
    t_min, t_max = np.amin(list_t), np.amax(list_t)

    tracker = Tracker(cfg.occup_ori_smooth_dist_th, 
                      cfg.occup_ori_smooth_max_frame_skip,
                      collect_states=True,
                      w_ori_tr=cfg.w_tr_pi)

    for dt in timerange(t_min, t_max):
      if dt in occup_dt[pi]:
        occup = occup_dt[pi][dt]

        ori = ori_dt[pi][dt]
        ori_norm = np.sqrt(np.sum(ori**2, axis=1)). reshape((-1,1))
        ori_norm[ori_norm==0] = 1.
        ori /= ori_norm
        
        # ori2d = ori_2d_dt[pi][dt]
        # ori_norm_2d = np.sqrt(np.sum(ori2d**2, axis=1)). reshape((-1,1))
        # ori_norm_2d[ori_norm_2d==0] = 1.
        # ori2d /= ori_norm_2d

        centers = np.concatenate([occup, ori], axis=1)
        
        tracker.update(centers, dt)
      else:
        tracker.update(ts=dt)
    tracker.close()

    for trackId in tracker.tracks_collected:

      ts = tracker.tracks_collected[trackId].ts
      center = tracker.tracks_collected[trackId].means_smoothed[:,:4]
      # print (center.shape)
      center = np.array(center)

      for i, dt in enumerate(ts):
        if dt not in occup_pi:
          occup_pi[dt] = {}
          ori_pi[dt] = {}
          #ori_pi_2d[dt] = {}

        if pi not in occup_pi[dt]:
          occup_pi[dt][pi] = np.empty((0,2))
          ori_pi[dt][pi] = np.empty((0,2))
          #ori_pi_2d[dt][pi] = np.empty((0,4))

        occup = center[i,:2].reshape((1,2))
        occup_pi[dt][pi] = np.concatenate((occup_pi[dt][pi], occup))

        ori = center[i,2:4].reshape((1,2))
        ori_norm = np.sqrt(np.sum(ori**2))
        if ori_norm == 0:
          ori_norm = 1.
        ori /= ori_norm

        ori_pi[dt][pi] = np.concatenate((ori_pi[dt][pi], ori))
        
        # ori2d = center[i,4:].reshape((1,4))
        # ori_norm_2d = np.sqrt(np.sum(ori2d**2))
        # if ori_norm_2d == 0:
        #   ori_norm_2d = 1.
        # ori2d /= ori_norm_2d

        # ori_pi_2d[dt][pi] = np.concatenate((ori_pi_2d[dt][pi], ori2d))
  
  return occup_pi, ori_pi
    
