import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def viz_occup_ori_ep6_pi_t(ori, occup, dt, pi, pid, cfg):

  file_ep6 = cfg.dir_data +'/ep6_floorplan_measured_half_gridded_1_meter.jpg'
  ep6_map = plt.imread(file_ep6)
  ep6_height, ep6_width, _ = ep6_map.shape

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
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/occup_ori_ep6'
  else:
    dir_copy_pid = dir_copy + f'/occup_ori_ep6/{pi}/{int(pid):03d}'
  os.makedirs(dir_copy_pid, exist_ok=True)

  file_copy_occup_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_occup_pi):
    x1, y1 = ori
    x2, y2 = occup

    # ori_end = ori + occup
    x3, y3 = x2 + x1, y2 + y1
    # # dir_copy_pid += '_occup+ori'
    
    # # ori_end = occup - ori
    # x3, y3 = x2 - x1, y2 - y1
    # dir_copy_pid += '_occup-ori'

    # # ori_end = occup - ori_swap
    # x3, y3 = x2 - y1, y2 - x1
    # dir_copy_pid += '_occup-ori_swap'

    # # ori_end = occup - ori-x+y
    # x3, y3 = x2 - x1, y2 + y1
    # dir_copy_pid += '_occup-ori-x+y'

    # os.makedirs(dir_copy_pid, exist_ok=True)

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(x2, y2, s=5, color='r', marker='o', alpha=0.7, label='occup')

    ax.plot([x2, x3], [y2, y3], color='b')

    # ax.legend(loc='upper right')

    # plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

def viz_occup_ori_ep6_pi(ori_ep6, EP6_feet_pos, dts, pi, cfg):
  for t in range(len(ori_ep6)):
    dt = dts[t]
    ori = ori_ep6[t]
    occup = EP6_feet_pos[t]
    viz_occup_ori_ep6_pi_t(ori, occup, dt, pi, pid, cfg)
