import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def viz_frame(dt_curr, pi, cfg):
  year = dt_curr.year
  month = dt_curr.month
  day = dt_curr.day
  hour = dt_curr.hour
  minute = dt_curr.minute
  second = dt_curr.second

  dir_pi = cfg.dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
  dir_vid = dir_pi + '/videos'

  if 0:
    ''' pre-minute grouping '''
    dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if cfg.use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_frame = dir_copy + '/frames'
  else:
    ''' per-data type grouping '''
    dir_copy_frame = cfg.dir_save + '/frames'
    
  dir_copy_frame_pi = dir_copy_frame + f'/{pi}'
  os.makedirs(dir_copy_frame_pi, exist_ok=True)

  file_frame_copy = dir_copy_frame_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
  if not os.path.exists(file_frame_copy):
    file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    frame = plt.imread(file_frame)
    print ('load from ...', file_frame)
    frame = np.flip(frame, axis=1)

    file_frame_copy = dir_copy_frame_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    plt.imsave(file_frame_copy, frame)
    print ('copy ...', file_frame, '... to ...', file_frame_copy)

