import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *

def viz_kps(dt_curr, pi, kps, folder_name, cfg):
  year = dt_curr.year
  month = dt_curr.month
  day = dt_curr.day
  hour = dt_curr.hour
  minute = dt_curr.minute
  second = dt_curr.second
  
  c_kps = cfg.cmap(np.linspace(0, 1, 10))    

  dir_pi = cfg.dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
  dir_vid = dir_pi + '/videos'

  if 0:
    ''' pre-minute grouping '''
    dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if cfg.use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_kps = dir_copy + f'/{folder_name}'
  else:
    ''' per-data type grouping '''
    dir_copy_kps = cfg.dir_save + f'/{folder_name}'
  dir_copy_kps_pi = dir_copy_kps + f'/{pi}'
  os.makedirs(dir_copy_kps_pi, exist_ok=True)

  file_copy_kps_pi = dir_copy_kps_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
  if not os.path.exists(file_copy_kps_pi):
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
    for i in range(kps.shape[0]):
      kp = kps[i]
      color = c_kps[i]

      for j1, j2 in connection:
        # print (j1, j2)
        # print (kp[j1-1], kp[j2-1])

        y1, x1 = kp[j1-1]
        y2, x2 = kp[j2-1]

        ax.plot(x1, y1, 'o', color=color, markersize=2)
        ax.plot(x2, y2, 'o', color=color, markersize=2)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_kps_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_kps_pi)
