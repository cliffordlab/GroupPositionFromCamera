import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from mpl_toolkits.mplot3d import Axes3D

def viz_pose3d_calib_match_2d_t(pos, pose2d, skeleton, axlim, dt, pi, pid, cfg):
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
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/pose3d_calib'
  else:
    dir_copy_pid = cfg.dir_save + f'/pose3d_calib/{pi}/{int(pid):03d}'
  os.makedirs(dir_copy_pid, exist_ok=True)

  file_save = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_save):
    # print (pose2d)
    # assert False

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=-90, azim=90)

    for p1, p2 in connection:
      col = 'red' if p1 in kps_right_2d else 'black'
      p1 -= 1
      p2 -= 1
      
      ax.plot(
        [pose2d[p1, 0], pose2d[p2, 0]],
        [pose2d[p1, 1], pose2d[p2, 1]],
        [0, 0],
        color=col)

    if axlim[0] > np.amin(pose2d):
      axlim[0] = np.amin(pose2d)
    if axlim[1] < np.amax(pose2d):
      axlim[1] = np.amax(pose2d)
    
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
            
        col = 'red' if j in skeleton.joints_right() else 'black'
        ax.plot([pose3d[j, 0], pose3d[j_parent, 0]],
                [pose3d[j, 1], pose3d[j_parent, 1]],
                [pose3d[j, 2], pose3d[j_parent, 2]], zdir='z', c=col)

    if axlim[0] > np.amin(pose3d):
      axlim[0] = np.amin(pose3d)
    if axlim[1] < np.amax(pose3d):
      axlim[1] = np.amax(pose3d)

    _idx_2d = np.array([12,14,16,11,13,15,5,7,9,6,8,10])
    _idx_3d = np.array([1,2,3,4,5,6,11,12,13,14,15,16])

    for i in range(len(_idx_2d)):
      # if i != 12:
      #   continue
      i_2d = _idx_2d[i]
      i_3d = _idx_3d[i]
      
      kp_2d = pose2d[i_2d]
      kp_3d = pose3d[i_3d]
      # print (i_2d, i_3d)
      # print (kp_2d, kp_3d)
      
      ax.plot(
        [kp_2d[0], kp_3d[0]],
        [kp_2d[1], kp_3d[1]],
        [0,        kp_3d[2]],
        color='b'
      )

    plt.tight_layout()

    plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_save)

def viz_pose3d_calib_match_2d(pose3d_calib, kps_seq, dts, pi, pid, cfg):

  sys.path.append(cfg.dir_code)
  from skeleton.h36m import Human36mDataset, Skeleton
  h36m = Human36mDataset()
  skeleton = h36m.skeleton()
  
  axlim = [np.amin(pose3d_calib), np.amax(pose3d_calib)]
  parents = skeleton.parents()
  for i, dt in enumerate(dts):
    pos = pose3d_calib[i]
    pose2d = kps_seq[i]
    viz_pose3d_calib_match_2d_t(pos, pose2d, skeleton, axlim, dt, pi, pid, cfg)
