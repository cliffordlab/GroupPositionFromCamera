import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def viz_pose3d_t(pos, skeleton, dt, axlim, pi, pid, cfg):
  parents = skeleton.parents()

  year = dt.year
  month = dt.month
  day = dt.day
  hour = dt.hour
  minute = dt.minute
  second = dt.second

  if 0:
    ''' pre-minute grouping '''
    dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if cfg.use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/pose3d'
  else:
    ''' per-data type grouping '''
    dir_copy_pid = cfg.dir_save + f'/pose3d/{pi}/{int(pid):03d}'
  os.makedirs(dir_copy_pid, exist_ok=True)

  file_save = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_save):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=125, azim=25)
    ax.set_xlim(axlim)
    ax.set_ylim(axlim)
    ax.set_zlim(axlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for j, j_parent in enumerate(parents):
      if j_parent == -1:
          continue
      # print (j, j_parent)
          
      col = 'red' if j in skeleton.joints_right() else 'black'
      ax.plot([pos[j, 0], pos[j_parent, 0]],
              [pos[j, 1], pos[j_parent, 1]],
              [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    plt.tight_layout()
    # plt.show()
    plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
    plt.close('all')
    print ('save in ...', file_save)

def viz_pose3d(prediction, dts, pi, pid, cfg):

  sys.path.append(cfg.dir_code)
  from skeleton.h36m import Human36mDataset, Skeleton
  h36m = Human36mDataset()
  skeleton = h36m.skeleton()
  
  axlim = np.amin(prediction), np.amax(prediction)
  for i, dt in enumerate(dts):
    pos = prediction[i]
    viz_pose3d_t(pos, skeleton, dt, axlim, pi, pid, cfg)