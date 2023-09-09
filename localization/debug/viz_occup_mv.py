import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def viz_occup_mv_t(occup, ori, dt, cfg):
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    if 0:
      dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
      if cfg.use_ep6_grid:
        dir_copy += '_grid'
      dir_copy += '_ori'
      dir_copy_occup = cfg.dir_save + f'/occup_mv_ov_{cfg.overlap_th}'
    else:
      dir_copy_occup = cfg.dir_save + f'/occup_mv_ov_{cfg.overlap_th}'
    os.makedirs(dir_copy_occup, exist_ok=True)

    file_copy_occup_pi = dir_copy_occup + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    if not os.path.exists(file_copy_occup_pi):
      
      fig, ax = plt.subplots()
      ax.imshow(ep6_map)
      ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7)

      # orientation
      for i in range(ori.shape[0]):
        ori_i = ori[i]
        occup_i = occup[i]
        x1, y1 = ori_i
        x2, y2 = occup_i
        
        x3, y3 = x2 + x1, y2 + y1
        ax.plot([x2, x3], [y2, y3], color='b')      

      plt.axis('off')
      plt.tight_layout()
      plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
      plt.close()
      print ('save in ...', file_copy_occup_pi)

def viz_occup_mv(occup_mv, ori_mv, cfg):
  ''' draw occupancy & orientation with multi-view '''
  for dt in occup_mv:
    occup = occup_mv[dt]
    ori = ori_mv[dt]
    viz_occup_mv_t(occup, ori, dt, cfg)

