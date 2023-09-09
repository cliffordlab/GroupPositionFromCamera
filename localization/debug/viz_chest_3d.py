import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def viz_chest_3d_t(pos, chest_start, chest_end, skeleton, dt, pi, folder_name, cfg):
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
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/chest_3d'
  else:
    dir_copy_pid = cfg.dir_save + f'/{folder_name}/{pi}/{int(pid):03d}'
  os.makedirs(dir_copy_pid, exist_ok=True)

  file_save = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_save):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim(axlim)
    ax.set_ylim(axlim)
    ax.set_zlim(axlim)
    ax.view_init(elev=125, azim=25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # pose 3d
    for j, j_parent in enumerate(parents):
      if j_parent == -1:
          continue
      # print (j, j_parent)
          
      # col = 'red' if j in skeleton.joints_right() else 'black'
      col='k'
      ax.plot([pos[j, 0], pos[j_parent, 0]],
              [pos[j, 1], pos[j_parent, 1]],
              [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
    
    left_sh = pos[11]
    x1, y1, z1 = left_sh
    right_sh = pos[14]
    x2, y2, z2 = right_sh
    if 1:
      spine = pos[7]
      x3, y3, z3 = spine
      thorax = pos[8]
      x4, y4, z4 = thorax
    else:
      left_hip = pos[4]
      x3, y3, z3 = left_hip
      right_hip = pos[1]
      x4, y4, z4 = right_hip
    ax.scatter(x1, y1, z1, s=5, color='r', marker='o', alpha=0.7, label='ori_start')
    ax.scatter(x2, y2, z2, s=5, color='b', marker='o', alpha=0.7, label='ori_start')
    ax.scatter(x3, y3, z3, s=5, color='g', marker='o', alpha=0.7, label='ori_start')
    ax.scatter(x4, y4, z4, s=5, color='g', marker='o', alpha=0.7, label='ori_start')

    # chest 3d
    x1, y1, z1 = chest_start
    x2, y2, z2 = chest_end
    # print (chest_start.shape, chest_end.shape)
    # assert False
    ax.plot([chest_start[0], chest_end[0]],
            [chest_start[1], chest_end[1]],
            [chest_start[2], chest_end[2]],
            zdir='z', c='blue')
    ax.scatter(x1, y1, z1, s=5, color='m', marker='o', alpha=0.7, label='ori_start')
    ax.scatter(x2, y2, z2, s=5, color='c', marker='o', alpha=0.7, label='ori_start')
    
    plt.tight_layout()
    plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_save)

def viz_chest_3d(pose3d_calib, chest_norm_3d, dts, pi, folder_name, cfg):
  sys.path.append(cfg.dir_code)
  from skeleton.h36m import Human36mDataset, Skeleton
  h36m = Human36mDataset()
  skeleton = h36m.skeleton()
  
  axlim = np.amin(pose3d_calib), np.amax(pose3d_calib)
  parents = skeleton.parents()
  for i, dt in enumerate(dts):
    pos = pose3d_calib[i]
    chest_start = chest_norm_3d[i,0,:]
    chest_end = chest_norm_3d[i,1,:]
    viz_chest_3d_t(pos, chest_start, chest_end, skeleton, dt, pi, folder_name, cfg)
    #-------------
