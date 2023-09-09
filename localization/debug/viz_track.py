import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def viz_track_t(mv_seq_dt, dt_curr, ep6_map, dir_copy_track_seconds, pid_color, cfg):
  # delta_minute = timedelta(minutes=1) # per-minute
  delta_second = timedelta(seconds=1) # per-minute

  year = dt_curr.year
  month = dt_curr.month
  day = dt_curr.day
  hour = dt_curr.hour
  minute = dt_curr.minute
  second = dt_curr.second

  file_save = dir_copy_track_seconds + f'/{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02}.png'

  if not os.path.exists(file_save):
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    dt_prev = dt_curr - delta_second  
    if dt_prev not in mv_seq_dt:
      return

    for pid, loc_curr, ori_curr in zip(mv_seq_dt[dt_curr]['pid'], 
                            mv_seq_dt[dt_curr]['loc'],
                            mv_seq_dt[dt_curr]['ori']
                            ):
      if pid not in mv_seq_dt[dt_prev]['pid']:
        continue
      idx = mv_seq_dt[dt_prev]['pid'].index(pid)
      loc_prev =  mv_seq_dt[dt_prev]['loc'][idx]
      ori_prev = mv_seq_dt[dt_prev]['ori'][idx]
      color = pid_color[pid]
      
      xs = [loc_prev[0], loc_curr[0]]
      ys = [loc_prev[1], loc_curr[1]]

      ax.plot(xs, ys, c='k')
      ax.scatter(loc_prev[0], loc_prev[1], s=5, color=color, marker='^', alpha=0.7, label=int(pid))
      ax.scatter(loc_curr[0], loc_curr[1], s=5, color=color, marker='^', alpha=0.7)

      # orientation
      # assert False
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)

def viz_track_pid(mv_seq, ep6_map, pid, dir_copy_track_pids, pid_color, cfg):

  file_save = dir_copy_track_pids + f'/id_{int(pid):03d}.png'
  if not os.path.exists(file_save):
    color = pid_color[pid]
    locs = np.array(mv_seq[pid]['loc']).reshape((-1, 2))
    oris = np.array(mv_seq[pid]['ori']).reshape((-1, 2))

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)

    # orientation
    for j in range(len(locs)):
      occup = locs[j]
      ori = oris[j]
      x1, y1 = ori
      x2, y2 = occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

    plt.savefig(file_save)
    plt.close()  
    print ('save in ...', file_save)

def viz_track(mv_seq, folder_name, cfg):

  file_ep6 = cfg.dir_data +'/ep6_floorplan_measured_half_gridded_1_meter.jpg'
  ep6_map = plt.imread(file_ep6)
  ep6_height, ep6_width, _ = ep6_map.shape
  
  list_pid = list(mv_seq.keys())
  list_pid.sort()
  c_list = cfg.cmap(np.linspace(0, 1, len(list_pid)))
  pid_color = {}
  for pid, color in zip(list_pid, c_list):
    pid_color[pid] = color

  dir_copy = cfg.dir_save + f'/{folder_name}'
  if cfg.use_ep6_grid:
    dir_copy += '_grid'
  dir_copy += '_ori'

  if 'kf' in folder_name:
    dir_copy_track = dir_copy
  else:    
    dir_copy_track = dir_copy + f'/track_th_{cfg.tracklet_th}'
    if 'fp' in folder_name:
      dir_copy_track += f'/_fp_{cfg.min_track_len}'
    if 'smth' in folder_name:
      dir_copy_track += f'/_smth_{cfg.smooth_track_len}'
    if 'ori' in folder_name:
      dir_copy_track += f'/_ori_track_w_{cfg.w_tr}'  
  os.makedirs(dir_copy_track, exist_ok=True)

  ''' plot all tracks on a single figure '''
  file_save = dir_copy_track + f'/all_Tracks.png'
  if True or not os.path.exists(file_save):
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    for pid in mv_seq:
      color = pid_color[pid]
      locs = np.array(mv_seq[pid]['loc']).reshape((-1, 2))
      oris = np.array(mv_seq[pid]['ori']).reshape((-1, 2))
      # print (locs.shape)
      # print (oris.shape)

      for j in range(len(locs)-1):
        ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=color)
      
      # orientation
      if 0:
        for j in range(len(locs)):
          occup = locs[j]
          ori = oris[j]
          x1, y1 = ori
          x2, y2 = occup
          x3, y3 = x2 + x1, y2 + y1
          ax.plot([x2, x3], [y2, y3], color='k')      

    plt.savefig(file_save)
    plt.close()  
    print ('save in ...', file_save)
  # assert False

  ''' draw tracking for each person '''  
  dir_copy_track_pids = dir_copy_track + f'/pIDs'  
  os.makedirs(dir_copy_track_pids, exist_ok=True)
  for pid in mv_seq:
    viz_track_pid(mv_seq, ep6_map, pid, dir_copy_track_pids, pid_color, cfg)

  ''' draw for each second for resulting tracks '''
  # each second
  mv_seq_dt = {}
  for pid in mv_seq:
    for dt_curr, loc, ori in zip(mv_seq[pid]['dt'],
                                mv_seq[pid]['loc'],
                                mv_seq[pid]['ori']):
      if dt_curr not in mv_seq_dt:
        mv_seq_dt[dt_curr] = {'pid': [], 
                              'loc': [],
                              'ori': []}
      mv_seq_dt[dt_curr]['pid'].append(pid)
      mv_seq_dt[dt_curr]['loc'].append(loc)
      mv_seq_dt[dt_curr]['ori'].append(ori)

  dir_copy_track_seconds = dir_copy_track + '/seconds'
  os.makedirs(dir_copy_track_seconds, exist_ok=True)

  for dt_curr in mv_seq_dt:
    viz_track_t(mv_seq_dt, dt_curr, ep6_map, dir_copy_track_seconds, pid_color, cfg)    