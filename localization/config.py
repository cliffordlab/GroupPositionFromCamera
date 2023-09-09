#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:11:04 2023

@author: chegde
"""

'''
Kalman filter combines:
1) False negative removal 
2) Tracking
3) Smoothing
So, only False positive removal needs to be done before or after.
'''

dir_code = '/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/localization'
dir_data = '/opt/scratchspace/chegde/EP6/data'
dir_posenet_data = '/labs/cliffordlab/data/EP6/openpose'
dir_exp = '/opt/scratchspace/chegde/EP6/exps'
dir_save = dir_exp + '/results'

frame_height = 481
frame_width = 641

use_ep6_grid = True

''' preprocessing '''
preprocessing_pose_method = 'kalman' # hungarian, kalman, particle
preproc_pose_dist_th = None
preproc_pose_max_frame_skip = 10

''' posenet '''
smooth_kps_len = 5 # sec

''' orientation '''
ori_method = 'mebow' # 2d, 3d, mebow

# MEBOW
mebow_model = 'gcn' # gcn, mlp, agcn
mebow_image = False,
path_mebow_repo = '/opt/scratchspace/chegde/EP6/repo/MEBOW'
path_mebow_cfg = path_mebow_repo +'/experiments/coco/segm-4_lr1e-3.yaml'
file_mebow_model = f'/opt/scratchspace/chegde/EP6/exps/mebow'
mebow_classification = True
cude_device_id = 0

# 3D method
path_videopose3d = '/opt/scratchspace/chegde/EP6/repo/VideoPose3D'
# path_videopose3d = r'C:\Users\hyeok.kwon\Research\Emory\EP6\repos\VideoPose3D'
chk_filename = '/opt/scratchspace/chegde/EP6/repo/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin'
# chk_filename = r'D:\Research\Emory\EP6\repos\VideoPose3D\model\pretrained_h36m_detectron_coco.bin'
smooth_calb_len = 5 # sec

''' Smooth location and orientation per pi'''
per_pi_occup_ori_smooth = 'kalman'# None, hungarian, kalman, particle
occup_ori_smooth_dist_th = 35
occup_ori_smooth_max_frame_skip = 10

w_tr_pi = 0. # moving direction weighting for orientation estimation

''' multi-view '''
# 1m: 10 (map) / 17.5 (grid)
overlap_th = 30
mv_camdist = False

''' tracking '''
# 1m: 10 (map) / 17.5 (grid)
tracking_mv_method = 'kalman' # hungarian, kalman, particle
mv_smooth_dist_th = 35
mv_smooth_max_frame_skip = 10

min_track_len = 60

tracklet_th = 20 # 1m: 10
tracklet_th = 35 # 1m: 10
smooth_track_len = 5

w_tr = 0. # moving direction weighting for orientation estimation

''' Grouping '''
# 1m: 10 (map) / 17.5 (grid)
det_group = False
interaction_th = 25
interaction_th = 43.5
interaction_ori = 180 # 180 degrees

''' Save & Load Intermediate '''
use_interm_kps_preproc_track = 1
use_interm_occup_pi = 0
use_interm_ori_pi = 1
use_interm_pose3d = 0
use_interm_ori3d_pnp = 0

''' Visualize '''
stop_after_viz = 0

viz_frame = 0

viz_kps = 0
viz_kps_fp1 = 0
viz_kps_smth_kf = 0
viz_kps_fp2 = 0
viz_kps_fn1 = 0
viz_kps_fn2 = 0
viz_kps_smth = 0

viz_kps_track = 0

viz_occup_ep6_pi = 0
viz_occup = 0

viz_mebow_img = 0

viz_pose3d = 0
viz_calib_proj = 0
viz_extrinsic = 0
viz_extrinsic_smth = 0
viz_pose3d_calib = 0
viz_pose3d_calib_match_2d = 0
viz_chest_3d = 0
viz_chest_3d_rot = 0
viz_chest_2d = 0
viz_chest_2d_cam = 0
viz_chest_2d_ep6 = 0
viz_occup_ori_ep6_pi = 0

viz_ori_2d = 0

viz_occup_ori = 0

viz_occup_mv_pi = 0
viz_occup_mv = 0
viz_occup_mv_fn1 = 0
viz_occup_mv_fn2 = 0

viz_tracking = 0
viz_tracking_fp = 0
viz_tracking_smth = 0
viz_tracking_ori = 0
viz_grouping = 0