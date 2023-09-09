'''--- Load data first ---'''

import sys
sys.path.append('localization')
from utils.tracking import timerange
from debug import viz_frame, viz_kps
from datetime import datetime

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(start_time, end_time, cfg):

  list_pi_ip = os.listdir(cfg.dir_proj_mat)
  list_pi_ip.sort()
  # pprint (list_pi_ip)

  list_pi_ip = [item[6:9] for item in list_pi_ip]
  # pprint (list_pi_ip)
  # assert False
  
  pi_data = {}
  list_dt = list(timerange(start_time, end_time))
  for t in range(len(list_dt)):
    dt_curr = list_dt[t]
    year = dt_curr.year
    month = dt_curr.month
    day = dt_curr.day
    hour = dt_curr.hour
    minute = dt_curr.minute
    second = dt_curr.second

    for pi_ip in list_pi_ip:
      pi = f'pi{pi_ip}.pi.bmi.emory.edu'
      
      dir_pi = cfg.dir_posenet_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_kps = dir_pi + '/keypoints'
      dir_vid = dir_pi + '/videos'

      file_kps = dir_kps + f'/{pi}{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}.npy'
      if not os.path.exists(file_kps):
        # print (file_kps, '... not exits')
        continue

      kps = np.load(file_kps)
      # print ('load from ...', file_kps)
      # print ('load from ...', file_kps, kps.shape)
          
      ''' collect frames '''
      if cfg.viz_frame:
        viz_frame(dt_curr, pi, cfg)
      ''''''
                  
      # remove zero keypoints
      kp_sum = np.sum(kps, axis=(1,2))
      kps = kps[kp_sum > 0,...]
      # print (kps.shape)
      if kps.shape[0] == 0: # remove empty keypoint
        continue
    
      # flip keypoints
      center_line = float(cfg.frame_width)/2
      kps[:,:,1] -= center_line
      kps[:,:,1] = -kps[:,:,1]
      kps[:,:,1] += center_line

      ''' draw kps on frames and save '''
      if cfg.viz_kps:
        viz_kps(dt_curr, pi, kps, 'kps', cfg)
      ''''''

      if pi not in pi_data:
        pi_data[pi] = {}

      key_dt = datetime(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second)

      pi_data[pi][key_dt] = kps

  if 0:
    for pi in pi_data:
      if 0:
        for key_dt in pi_data[pi]:
          print (pi, key_dt)
      else:
        print (pi, len(pi_data[pi].keys()))
      print ('-------------------')

  if cfg.stop_after_viz:
    if cfg.viz_frame | cfg.viz_kps:
      assert False

  return pi_data