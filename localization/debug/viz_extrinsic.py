import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

def viz_extrinsic(calib_3d, dt, pi, pid, is_smth, cfg):

  if 0:
    ''' pre-minute grouping '''
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if cfg.use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_pid = dir_copy + f'/orientation/{pi}/{int(pid):03d}'
  else:
    ''' per-data type grouping '''
    dir_copy_pid = cfg.dir_save + f'/extrinsics/{pi}/{int(pid):03d}'

  os.makedirs(dir_copy_pid, exist_ok=True)

  file_copy_kps_fp_rm_pi = dir_copy_pid
  if is_smth:
    file_copy_kps_fp_rm_pi += '_smth'
  file_copy_kps_fp_rm_pi += '.jpg'

  if not os.path.exists(file_copy_kps_fp_rm_pi):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    # rotation
    proj_rot = calib_3d['rot']
    xs = np.arange(proj_rot.shape[0])
    axes[0].plot(xs, proj_rot[:,0], c='r', label='rot_x')
    axes[0].plot(xs, proj_rot[:,1], c='g', label='rot_y')
    axes[0].plot(xs, proj_rot[:,2], c='b', label='rot_z')
    axes[0].legend(loc='upper right')
    # translation
    proj_trans = calib_3d['trans']
    xs = np.arange(proj_trans.shape[0])
    axes[1].plot(xs, proj_trans[:,0], c='r', label='trans_x')
    axes[1].plot(xs, proj_trans[:,1], c='g', label='trans_y')
    axes[1].plot(xs, proj_trans[:,2], c='b', label='trans_z')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(file_copy_kps_fp_rm_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_kps_fp_rm_pi)
