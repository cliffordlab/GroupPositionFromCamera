import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *

def viz_calib_proj_t(pose2d, pose3d, rotation_vector, translation_vector, dt, skeleton, cfg):

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
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}/calib_proj'
  else:
    ''' per-data type grouping '''
    dir_copy_pid = cfg.dir_save + f'/calib_proj/{pi}/{int(pid):03d}'
  os.makedirs(dir_copy_pid, exist_ok=True)

  file_copy_kps_fp_rm_pi = dir_copy_pid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_kps_fp_rm_pi):

    imagePoints, jacobian = cv2.projectPoints(
                              objectPoints=pose3d, 
                              rvec=rotation_vector, 
                              tvec=translation_vector, 
                              cameraMatrix=camera_matrix, 
                              distCoeffs=dist_coeffs)
    imagePoints = imagePoints.reshape((imagePoints.shape[0], 2))
    # print (pose2d.shape)
    # print (pose3d.shape)
    # print (rotation_vector.shape)
    # print (translation_vector.shape)
    # print (imagePoints.shape)
    # assert False

    dir_pi = cfg.dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
    dir_vid = dir_pi + '/videos'
    file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
    if os.path.exists(file_frame):
      frame = plt.imread(file_frame)
      # print (frame.shape)
      # assert False
      # print ('load from ...', file_frame)
      frame = np.flip(frame, axis=1)
    else:
      frame = np.ones((cfg.frame_height, cfg.frame_width, 3))

    fig, ax = plt.subplots()
    ax.imshow(frame)

    kp = pose2d
    color = 'r'

    for j1, j2 in connection:
      # print (j1, j2)
      # print (kp[j1-1], kp[j2-1])

      y1, x1 = kp[j1-1]
      y2, x2 = kp[j2-1]

      ax.plot(x1, y1, 'o', color=color, markersize=2)
      ax.plot(x2, y2, 'o', color=color, markersize=2)
      ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)

    kp = imagePoints
    color = 'b'

    parents = skeleton.parents()
    for j, j_parent in enumerate(parents):
      if j_parent == -1:
          continue
      # print (j, j_parent)
      y1, x1 = kp[j]
      y2, x2 = kp[j_parent]

      ax.plot(x1, y1, 'o', color=color, markersize=2)
      ax.plot(x2, y2, 'o', color=color, markersize=2)
      ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_kps_fp_rm_pi)

def viz_calib_proj(kps_seq, prediction, dts, calib_3d, cfg):

  sys.path.append(cfg.dir_code)
  from skeleton.h36m import Human36mDataset, Skeleton
  h36m = Human36mDataset()
  skeleton = h36m.skeleton()

  for t in range(kps_seq.shape[0]):
    dt = dts[t]
    pose2d = kps_seq[t]
    pose3d = prediction[t]
    rotation_vector = calib_3d['rot'][t]
    translation_vector = calib_3d['trans'][t]

    viz_calib_proj_t(pose2d, pose3d, rotation_vector, translation_vector, dt, skeleton, cfg)