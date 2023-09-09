'''
1. Track with forward pass
2. Remove short tracks
3. Backward Smooth
'''

from tracking.kf.kf_pose import Tracker
from utils.tracking import timerange
import numpy as np
from debug import viz_kps_track, viz_kps

def kps_smth_kf(pi_data, cfg):

  #------------------------
  # Track with forward pass
  #-------------------------

  pi_data_smth = {}
  pi_pid_seq = {}
  kps_seq_pi_pid_np = {}
  for pi in pi_data:
    list_t = list(pi_data[pi])
    t_min, t_max = np.amin(list_t), np.amax(list_t) 

    tracker = Tracker(cfg.preproc_pose_dist_th, 
                      cfg.preproc_pose_max_frame_skip, 
                      collect_states=True)

    for dt in timerange(t_min, t_max):
      if dt in pi_data[pi]:
        kps_det = pi_data[pi][dt]
        # print (kps_det.shape)
        kps_det = kps_det.reshape((-1, 34))
        # print (kps_det.shape)
        # assert False
        tracker.update(kps_det, dt)
      else:
        tracker.update(ts=dt)
    tracker.close()

    pi_data_smth[pi] = {}
    pi_pid_seq[pi] = {}
    kps_seq_pi_pid_np[pi] = {}
    for trackId in tracker.tracks_collected:

      ts = tracker.tracks_collected[trackId].ts
      kps_smth = tracker.tracks_collected[trackId].means_smoothed
      kps_smth = np.array(kps_smth)
      # print (kps_smth.shape)
      kps_smth = kps_smth[:,:34].reshape((-1, 17, 2))
      # print (len(ts), kps_smth.shape)
      # assert False

      if trackId not in pi_pid_seq[pi]:
        pi_pid_seq[pi][trackId] = {'dt': [], 'rid': []}

      kps_seq_pi_pid_np[pi][trackId] = kps_smth        

      for i, dt in enumerate(ts):
        if dt not in pi_data_smth[pi]:
          pi_data_smth[pi][dt] = np.empty((0, 17, 2))

        # print (kps_smth[i].shape)
        # assert False
                
        pi_data_smth[pi][dt] = np.concatenate((pi_data_smth[pi][dt], kps_smth[i].reshape(1, 17, 2)))

        pi_pid_seq[pi][trackId]['dt'].append(dt)
        pi_pid_seq[pi][trackId]['rid'].append(pi_data_smth[pi][dt].shape[0]-1)
  
  pi_data = pi_data_smth

  ''' draw: compare between detection VS filtered VS smoothed '''
  if cfg.viz_kps_smth_kf:
    for pi in pi_data:
      for dt in pi_data[pi]:
        kps = pi_data[pi][dt]
        viz_kps(dt, pi, kps, 'kps_smth_kf', cfg)
    # assert False
    
    for pi in kps_seq_pi_pid_np:
      for pid in kps_seq_pi_pid_np[pi]:
        dts = pi_pid_seq[pi][pid]['dt']
        kps_seq = kps_seq_pi_pid_np[pi][pid]
        viz_kps_track(kps_seq, dts, pi, pid, 'kps_smth_kf_track', cfg)
    # assert False

  return pi_data, pi_pid_seq, kps_seq_pi_pid_np