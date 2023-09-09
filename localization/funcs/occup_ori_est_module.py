'''
1) Orientation Estimation
- a. Generate 3D pose
- b. PnP 2D & 3D pose
- c. smooth calibrated orientations
- d. calibrate 3D pose
- e. Get 3D chest normal vector
- f. Project on the x-y plane
- g. Reflect camera direction in EP6
- h. Project camera-reflected orientation to EP6 axis
- i. project foot location
'''

import numpy as np
import torch
import cv2
import sys
import os
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
sys.path.append('/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/localization')
from utils.tracking import *
import copy
from pandas import Series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.occup_ori import *
from debug import viz_occup_ori
from funcs.ori_est import ori_mebow_est, ori_2d_est
#from funcs import *

def occup_ori_est(pi_data, pi_pid_seq,
                  kps_seq_pi_pid_np, cfg):

  # dir_result = cfg.dir_save + '/occup'
  # os.makedirs(dir_result, exist_ok=True)
  # file_result = dir_result + '/occup_pi.p'
  # if cfg.use_interm_occup_pi \
  #   and os.path.exists(file_result):
  #   occup_pi = cp.load(open(file_result, 'rb'))
  #   print ('load from ...', file_result)
  # else:
  ''' i. Project Foot to EP6 map '''
  print ('1) Project Foot to EP6 map')
  EP6_feet_pos_pi_pid = proj_foot_ep6(kps_seq_pi_pid_np, pi_pid_seq, cfg)

  ''' collect occupancy '''
  print ('2) Collect occupancy ')
  occup_pi = collect_occup(EP6_feet_pos_pi_pid, pi_pid_seq, pi_data, cfg)
    # if 0:
    #   return 0, 0

    # cp.dump(occup_pi, open(file_result, 'wb'))
    # print ('save in ...', file_result)
  
  '''
  Orientation estimation
  '''
  print ('3) Orientation Estimation-mebow')
  # dir_result = cfg.dir_save + f'/ori'
  # os.makedirs(dir_result, exist_ok=True)
  # file_result = dir_result + '/ori_pi.p'
  # if cfg.use_interm_ori_pi and os.path.exists(file_result):
  #   ori_pi = cp.load(open(file_result, 'rb'))
  #   print ('load from ...', file_result)
  # else:  
  ori_pi_mebow = ori_mebow_est(pi_pid_seq, kps_seq_pi_pid_np,
                      EP6_feet_pos_pi_pid, cfg)
  
  ori_pi_2d = ori_2d_est(pi_pid_seq, kps_seq_pi_pid_np, cfg)
    # cp.dump(ori_pi, open(file_result, 'wb'))
    # print ('save in ...', file_result)

  # if cfg.viz_occup_ori:
  #   ''' draw occupancy & Orientation observed from all Pis '''
  #   viz_occup_ori(ori_pi, occup_pi, pi_data, cfg) 
  #   assert False   

  # if cfg.stop_after_viz:
  #   if cfg.viz_kps_track | cfg.viz_pose3d | cfg.viz_calib_proj | cfg.viz_extrinsic \
  #     | cfg.viz_extrinsic_smth | cfg.viz_pose3d_calib \
  #     | cfg.viz_pose3d_calib_match_2d | cfg.viz_chest_3d \
  #     | cfg.viz_chest_2d | cfg.viz_chest_2d_cam | cfg.viz_chest_2d_ep6 \
  #     | cfg.viz_occup_ori_ep6_pi:
  #     assert False
  print('Finished')
  return occup_pi, ori_pi_mebow, ori_pi_2d