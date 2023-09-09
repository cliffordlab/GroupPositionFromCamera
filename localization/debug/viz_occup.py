import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def viz_occup_t_pi(occup, dt, pi, cfg):

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
    dir_copy_occup = dir_copy + '/occup'
  else:
    dir_copy_occup = cfg.dir_save + '/occup'
  dir_copy_occup_pi = dir_copy_occup + f'/{pi}'
  os.makedirs(dir_copy_occup_pi, exist_ok=True) 

  file_copy_occup_pi = dir_copy_occup_pi + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_occup_pi):

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for i in range(occup.shape[0]):
      color = c_kps[i]
      # print (color)
      occup_ = occup[i]
      # print (occup_)
      x2, y2 = occup_

      ax.scatter(x2, y2, s=5, color='r', marker='o', alpha=0.7, label='occup')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

def viz_occup_t(occup_pi_dt, pi_color, dt, cfg):

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
    dir_copy_occup = dir_copy + '/occup'
  else:
    dir_copy_occup = cfg.dir_save + '/occup'
  dir_copy_occup_indiv = dir_copy_occup + '/individual'
  os.makedirs(dir_copy_occup_indiv, exist_ok=True)

  file_copy_occup_pi = dir_copy_occup_indiv + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_occup_pi):
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for pi in occup_pi_dt:
          
      occup = occup_pi_dt[pi]
      # print (occup.shape)
      # assert False
      color = pi_color[pi]

      ax.scatter(occup[:,0], occup[:,1], s=10, color=color, marker='o', alpha=0.7, label=pi)
      
    ax.legend()

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

def viz_occup(occup_pi, pi_data, cfg):
  ''' draw occupancy & Orientation observed from all Pis '''
  
  list_pi = list(pi_data.keys())
  list_pi.sort()
  c_pis = cfg.cmap(np.linspace(0, 1, len(list_pi)))
  pi_color = {}
  for pi, c in zip(list_pi, c_pis):
    pi_color[pi] = c

  # per pi
  for dt in occup_pi:
    for pi in occup_pi[dt]:
      occup = occup_pi[dt][pi]
      # assert False
      viz_occup_t_pi(occup, dt, pi, cfg)

  # all pis
  for dt in occup_pi:
    occup_pi_dt = occup_pi[dt]
    viz_occup_t(occup_pi_dt, pi_color, dt, cfg)

