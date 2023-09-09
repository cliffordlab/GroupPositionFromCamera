import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *

def viz_kps_t(kp, dt, pi, pid, dir_name, cfg):
  year = dt.year
  month = dt.month
  day = dt.day
  hour = dt.hour
  minute = dt.minute
  second = dt.second

  dir_pi = cfg.dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
  dir_vid = dir_pi + '/videos'

  if 0:
    ''' pre-minute grouping '''
    dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if cfg.use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/{dir_name}'
  else:
    ''' per-data type grouping '''
    dir_copy_pid = cfg.dir_save + f'/{dir_name}/{pi}/{int(pid):03d}'

  os.makedirs(dir_copy_pid, exist_ok=True)

  file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_kps_fp_rm_pi):
    file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    if os.path.exists(file_frame):
      frame = plt.imread(file_frame)
      # print (frame.shape)
      # assert False
      # print ('load from ...', file_frame)
      frame = np.flip(frame, axis=1)
    else:
      frame = np.ones((cfg.frame_height, cfg.frame_width, 3))
    
    fig, ax = plt.subplots()
    ax.imshow(frame)

    color = 'k'

    for j1, j2 in connection:
      # print (j1, j2)
      # print (kp[j1-1], kp[j2-1])

      y1, x1 = kp[j1-1]
      y2, x2 = kp[j2-1]

      ax.plot(x1, y1, 'o', color=color, markersize=2)
      ax.plot(x2, y2, 'o', color=color, markersize=2)
      ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)

    # left shoulder -> Actually right shoulder (video flip error)
    y1, x1 = kp[5]
    ax.plot(x1, y1, 'o', color='r', markersize=5)
    # right shoulder -> Actually left shoulder (video flip error)
    y1, x1 = kp[6]
    ax.plot(x1, y1, 'o', color='b', markersize=5)              
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_kps_fp_rm_pi)

def viz_kps_track(kps_seq, dts, pi, pid, dir_name, cfg):
  for i, dt in enumerate(dts):
    kp = kps_seq[i]
    viz_kps_t(kp, dt, pi, pid, dir_name, cfg)
