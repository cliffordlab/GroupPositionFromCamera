import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def viz_occup_ori_t_pi(ori, occup, dt, pi, cfg):

  file_ep6 = cfg.dir_data +'/ep6_floorplan_measured_half_gridded_1_meter.jpg'
  ep6_map = plt.imread(file_ep6)
  ep6_height, ep6_width, _ = ep6_map.shape
  
  c_kps = cfg.cmap(np.linspace(0, 1, 10))    
  
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
    dir_copy_occup = dir_copy + '/occup_ori'
  else:
    dir_copy_occup = cfg.dir_save + '/occup_ori'
  dir_copy_occup_pi = dir_copy_occup + f'/{pi}'
  os.makedirs(dir_copy_occup_pi, exist_ok=True) 

  file_copy_occup_pi = dir_copy_occup_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if True or not os.path.exists(file_copy_occup_pi):

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for i in range(occup.shape[0]):
      color = c_kps[i]
      # print (color)
      occup_ = occup[i]
      # print (occup_)
      ori_ = ori[i]
      # print (ori_)
      # assert False

      x1, y1 = ori_*10
      x2, y2 = occup_
      x3, y3 = x2 + x1, y2 + y1

      ax.scatter(x2, y2, s=5, color='r', marker='o', alpha=0.7, label='occup')

      ax.plot([x2, x3], [y2, y3], color='b')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

def viz_occup_ori_t(ori_pi_dt, occup_pi_dt, pi_color, cfg):

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
    dir_copy_occup = dir_copy + '/occup_ori'
  else:
    dir_copy_occup = cfg.dir_save + '/occup_ori'
  dir_copy_occup_indiv = dir_copy_occup + '/individual'
  os.makedirs(dir_copy_occup_indiv, exist_ok=True)

  file_copy_occup_pi = dir_copy_occup_indiv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if True or not os.path.exists(file_copy_occup_pi):
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for pi in occup_pi[dt]:
          
      occup = occup_pi[dt][pi]
      # print (occup.shape)
      ori_start = ori_pi[dt][pi]
      # print (ori_start.shape)
      ori_end = ori_start + occup
      # assert False
      color = pi_color[pi]

      ax.scatter(occup[:,0], occup[:,1], s=10, color=color, marker='o', alpha=0.7, label=pi)
      
      for i in range(ori_start.shape[0]):
        x1, y1 = ori_start[i]
        x2, y2 = occup[i]*10
        x3, y3 = x2 + x1, y2 + y1
        ax.plot([x2, x3], [y2, y3], color=color)

    ax.legend()

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

def viz_occup_ori(ori_pi, occup_pi, pi_data, cfg):
  ''' draw occupancy & Orientation observed from all Pis '''
  
  list_pi = list(pi_data.keys())
  list_pi.sort()
  c_pis = cfg.cmap(np.linspace(0, 1, len(list_pi)))
  pi_color = {}
  for pi, c in zip(list_pi, c_pis):
    pi_color[pi] = c

  # per pi
  for dt in ori_pi:
    for pi in ori_pi[dt]:
      occup = occup_pi[dt][pi]
      ori = ori_pi[dt][pi]
      # print (ori)
      # print (ori.shape)
      # assert False
      viz_occup_ori_t_pi(ori, occup, dt, pi, cfg)

  # all pis
  for dt in ori_pi:
    occup_pi_dt = occup_pi[dt]
    ori_pi_dt = ori_pi[dt]
    viz_occup_ori_t(ori_pi_dt, occup_pi_dt, pi_color, cfg)

