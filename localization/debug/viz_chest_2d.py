import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def viz_chest_2d_t(chest_2d, dt, pi, pid, folder_name, cfg):
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
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/{folder_name}'
  else:
    dir_copy_pid = cfg.dir_save + f'/{folder_name}/{pi}/{int(pid):03d}'
  os.makedirs(dir_copy_pid, exist_ok=True)

  file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_kps_fp_rm_pi):
    fig, ax = plt.subplots()
    x, y = chest_2d
    ax.plot([0, x], [0, y], 'b')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_kps_fp_rm_pi)

def viz_chest_2d(chest_2d_vec, dts, pi, pid, folder_name, cfg):
  for i, dt in enumerate(dts):
    chest_2d = chest_2d_vec[i]
    viz_chest_2d_t(chest_2d, dt, pi, pid, folder_name, cfg)
