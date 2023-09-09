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
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
import copy
from pandas import Series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from debug import viz_kps_track, viz_pose3d, viz_calib_proj, viz_extrinsic, viz_pose3d_calib_match_2d, viz_chest_3d, viz_chest_2d, viz_occup_ori_ep6_pi, viz_occup_ori

def occup_ori_est(pi_data, cfg):
  # prepare 3D lifting model
  # sys.path.append(r'C:\Users\hyeok.kwon\Research\Emory\EP6\repos\VideoPose3D')
  sys.path.append('/home/hhyeokk/Research/EP6/repo/VideoPose3D')

  file_ep6 = cfg.dir_data +'/ep6_floorplan_measured_half_gridded_1_meter.jpg'
  ep6_map = plt.imread(file_ep6)
  ep6_height, ep6_width, _ = ep6_map.shape

  coco_metadata = {
      'layout_name': 'coco',
      'num_joints': 17,
      'keypoints_symmetry': [
          # 0: nose
          [1, # left eye
          3, # left ear
          5, # left shoulder
          7,  # left elbow
          9,  # left wrist
          11, # left hip
          13, # left knee
          15], # left ankle
          [2, # right eye
          4, # right ear
          6, # right shoulder
          8, # right elbow
          10, # right wrist
          12, # right hip
          14, # right knee
          16], # right ankle
      ]
  }

  metadata = coco_metadata
  metadata['video_metadata'] = {'temp_name': {'h': cfg.frame_height, 'w': cfg.frame_width}}
  keypoints_symmetry = metadata['keypoints_symmetry']
  kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
  from skeleton.h36m import Human36mDataset, Skeleton
  h36m = Human36mDataset()
  skeleton = h36m.skeleton()
  joints_left, joints_right = list(h36m.skeleton().joints_left()), list(h36m.skeleton().joints_right())

  from common.model import TemporalModel
  architecture = '3,3,3,3,3'
  filter_widths = [int(x) for x in architecture.split(',')]
  causal = False
  dropout = 0.25
  channels = 1024
  dense = False
  model_pos = TemporalModel(
    metadata['num_joints'], 
    2, 
    metadata['num_joints'],
    filter_widths=filter_widths, 
    causal=causal, 
    dropout=dropout, 
    channels=channels,
    dense=dense)

  receptive_field = model_pos.receptive_field()
  # print('INFO: Receptive field: {} frames'.format(receptive_field))
  pad = (receptive_field - 1) // 2 # Padding on each side
  if causal:
      # print('INFO: Using causal convolutions')
      causal_shift = pad
  else:
      causal_shift = 0

  assert torch.cuda.is_available()
  device = torch.device('cuda')
  model_pos = model_pos.to(device)

  checkpoint = torch.load(cfg.chk_filename, map_location=lambda storage, loc: storage)
  epoch = checkpoint['epoch']
  print(f'This model was trained for {epoch} epochs')
  model_pos.load_state_dict(checkpoint['model_pos'])

  from common.camera import normalize_screen_coordinates
  from scipy.spatial.transform import Rotation as R

  # prepare camera matrices
  size = [cfg.frame_height, cfg.frame_width]
  focal_length = size[1]
  center = (size[1]/2, size[0]/2)
  camera_matrix = np.array(
                          [[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], 
                          dtype = "double")
  dist_coeffs = np.zeros((4,1)) # no lens distortion

  # chest keypoints
  #     {12, "RHip"},
  #     {11, "LHip"},
  #     {5,  "LShoulder"},
  #     {6,  "RShoulder"},
  idx_2d = np.array([12,11,5,6])
  # 'RightHip': 1,          # Hip.R
  # 'LeftHip': 4,           # Hip.L
  # 'LeftShoulder': 11,     # Shoulder.L
  # 'RightShoulder': 14,    # Shoulder.R
  idx_3d = np.array([1,4,11,14])

  solvepnp_flag = cv2.SOLVEPNP_EPNP # So this gives the best improvment

  #--------------------------
  dir_proj_mat = cfg.dir_exp + '/proj_mat'

  ori_pi = {}
  occup_pi = {}
  for pi in pi_data:

    pi_ip = int(pi[2:5])
    file_save = dir_proj_mat +f'/pi_ip_{pi_ip}.npy'
    assert os.path.exists(file_save), pi_ip
    if not os.path.exists(file_save):
      print (file_save, '... not exist')
      continue
    M = np.load(file_save)
    
    list_dt = list(pi_data[pi].keys())
    list_dt.sort()
    list_ts = [datetime.timestamp(item) for item in list_dt]

    max_pid = None
    for k, g in groupby(enumerate(list_ts), lambda ix: ix[0]-ix[1]):
      block_ts = list(map(itemgetter(1), g))
      block_dt = [datetime.fromtimestamp(item) for item in block_ts]

      # track keypoints in the block
      pid_seq = {}
      tracker_kps = hungarian_kps(max_pid=max_pid)
      for i in range(len(block_dt)):
        dt_curr = block_dt[i]
        kps_curr = pi_data[pi][dt_curr]

        try:
          pid = tracker_kps.update(kps_curr)
        except:
          print (pi, dt_curr)
          print (kps_curr)
          assert False

        for rid, pid in enumerate(pid):
          if pid not in pid_seq:
            pid_seq[pid] = {'dt': [], 'rid': []}
          if max_pid is None:
            max_pid = pid
          elif max_pid < pid:
            max_pid = pid
            
          pid_seq[pid]['dt'].append(dt_curr)
          pid_seq[pid]['rid'].append(rid)
      
      for pid in pid_seq:
        dts = pid_seq[pid]['dt']
        rids = pid_seq[pid]['rid']
        if len(dts)> 1:

          # collect sequence
          kps_seq = np.empty((0, 17, 2))
          for dt, rid in zip(dts, rids):
            kps = pi_data[pi][dt][rid].reshape((1, 17, 2))
            kps_seq = np.concatenate((kps_seq, kps))
          
          if cfg.viz_kps_track:
            viz_kps_track(kps_seq, dts, pi, pid, cfg)
          
          ''' a. lift 2D -> 3D '''
          kps_seq_norm = copy.deepcopy(kps_seq)
          kps_seq_norm[..., :2] = normalize_screen_coordinates(kps_seq_norm[..., :2], w=cfg.frame_width, h=cfg.frame_height)
          with torch.no_grad():
            model_pos.eval()
            # print (kps_seq_norm.shape)
            # assert False

            batch_2d = np.expand_dims(np.pad(kps_seq_norm[..., :2],
                        ((pad + causal_shift, pad - causal_shift), 
                        (0, 0), (0, 0)), 'edge'), axis=0)

            batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
            batch_2d[1, :, :, 0] *= -1
            batch_2d[1, :, kps_left + kps_right] = batch_2d[1, :, kps_right + kps_left]

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.to(device)
            inputs_2d = inputs_2d.contiguous()
            # print (inputs_2d.shape)
            # assert False
            predicted_3d_pos = model_pos(inputs_2d)

            # Undo flipping and take average with non-flipped version
            predicted_3d_pos[1, :, :, 0] *= -1
            predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
            predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            # print ('predicted_3d_pos:', predicted_3d_pos.size())
            prediction = predicted_3d_pos.squeeze(0).cpu().numpy()  
          
          if cfg.viz_pose3d:
            viz_pose3d(prediction, skeleton, dts, cfg)

          ''' b. PnP 2D & 3d '''
          calib_3d = {'rot': [], 'trans': []}
          for t in range(kps_seq.shape[0]):
            pose2d = kps_seq[t]
            pose3d = prediction[t]
            pose2D_pair = pose2d[idx_2d].astype(np.float32)
            pose3D_pair = pose3d[idx_3d].astype(np.float32)
            # print (pose2D_pair)
            # print(pose2D_pair.shape, type(pose2D_pair), pose2D_pair.dtype)
            # print (pose3D_pair)
            # print(pose3D_pair.shape, type(pose3D_pair), pose3D_pair.dtype)
            # print ('---')

            r = np.array([], dtype=np.float32)

            success, rotation_vector, translation_vector, reprojectionError = cv2.solvePnPGeneric(
                              objectPoints=pose3D_pair,
                              imagePoints=pose2D_pair, 
                              cameraMatrix=camera_matrix, 
                              distCoeffs=dist_coeffs, 
                              useExtrinsicGuess=False,
                              reprojectionError=r,
                              flags=solvepnp_flag)
            reprojectionError = reprojectionError.reshape((-1,))
            # print (len(rotation_vector))
            # print (rotation_vector[0])
            # print (len(translation_vector))
            # print (translation_vector[0])
            # print (reprojectionError.shape)
            # assert False
            sel = np.argmin(reprojectionError)
            rotation_vector = rotation_vector[sel]
            translation_vector = translation_vector[sel]

            calib_3d['rot'].append(rotation_vector)
            calib_3d['trans'].append(translation_vector)
          
          calib_3d['rot'] = np.array(calib_3d['rot'])
          calib_3d['trans'] = np.array(calib_3d['trans'])

          if cfg.viz_calib_proj:
            viz_calib_proj(kps_seq, prediction, skeleton, dts, calib_3d, cfg)

          if cfg.viz_extrinsic:
            dt = dts[0]
            viz_extrinsic(calib_3d, dt, pi, False, cfg)
          
          ''' c. smooth calibrated 1) rotation & 2) translation ''' 
          # 1) smooth rotation
          proj_rot = calib_3d['rot']
          proj_rot = proj_rot.reshape((proj_rot.shape[0],3))
          rot = R.from_rotvec(proj_rot)
          quat = rot.as_quat()
          for ch in range(quat.shape[1]):
            quat_ch = quat[:,ch]
            quat_ch = Series(quat_ch).rolling(cfg.smooth_calb_len, min_periods=1, center=True).mean().to_numpy()
            quat[:,ch] = quat_ch
          rot = R.from_quat(quat)
          proj_rot = rot.as_rotvec()
          
          calib_3d['rot'] = proj_rot.reshape((proj_rot.shape[0],3,1))
          
          # smoothe trans
          proj_trans = calib_3d['trans']
          proj_trans = proj_trans.reshape((proj_trans.shape[0],3))
          for ch in range(proj_trans.shape[1]):
            trans_ch = proj_trans[:,ch]
            trans_ch = Series(trans_ch).rolling(cfg.smooth_calb_len, min_periods=1, center=True).mean().to_numpy()
            proj_trans[:,ch] = trans_ch
          
          calib_3d['trans'] = proj_trans.reshape((proj_trans.shape[0],3,1))

          if cfg.viz_extrinsic_smth:
            dt = dts[0]
            viz_extrinsic(calib_3d, dt, pi, True, cfg)

          ''' d. calibrate 3d pose 
          The calibrated 3D pose represents the 3D pose view from camera.
          So, the chest vector should be okay to reflect where the person is looking at.
          The noise will be smoothed temporally.
          '''
          
          pose3d_calib = []
          for t in range(prediction.shape[0]):
            pose3d = prediction[t]
            rotation_vector = calib_3d['rot'][t]
            translation_vector = calib_3d['trans'][t]

            cMo = np.eye(4,4)
            Rot = cv2.Rodrigues(rotation_vector)[0]
            cMo[0:3,0:3] = Rot
            cMo[0:3,3] = translation_vector.reshape((3,))

            Rot = cMo[:3,:3]
            if 0:
              Rot = Rot.T
            Trans = cMo[:3,3].reshape((3,1))
            
            if 0: # for orientation translation is not really needed.
              pose3d = np.dot(Rot, pose3d.T)
            else:
              pose3d = np.dot(Rot, pose3d.T) + Trans
            pose3d = pose3d.T
            # pose3d = -pose3d # Need to flip!!!

            pose3d_calib.append(pose3d)

          pose3d_calib = np.array(pose3d_calib)

          if cfg.viz_pose3d_calib_match_2d:
            viz_pose3d_calib_match_2d(pose3d_calib, kps_seq, skeleton, dts, pi, cfg)        
                
          ''' e. Get 3D chest normal vector '''
          left_sh = pose3d_calib[:, 11]
          right_sh = pose3d_calib[:, 14]
          spine = pose3d_calib[:,7]
          thorax = pose3d_calib[:,8]
          left_hip = pose3d_calib[:, 4]
          # print (left_sh.shape)
          # print (right_sh.shape)
          # print (left_hip.shape)
          hori_axis = right_sh - left_sh
          if 1:
            vert_axis = spine - thorax
            chest_start = np.sum(pose3d_calib[:,[11,14,7,8]], axis=1)/4
            # print (pose3d_calib[:,idx_3d].shape)
            # print (chest_start.shape)
            chest_start = chest_start.reshape((-1,1,3))
            # print (chest_start.shape)
          else:
            assert False, 'left and right hip is not on the same plane as there is one more joint (spine) in the middle'
            vert_axis = left_hip - left_sh
            chest_start = np.sum(pose3d_calib[:,idx_3d], axis=1)/4
            # print (pose3d_calib[:,idx_3d].shape)
            # print (chest_start.shape)
            chest_start = chest_start.reshape((-1,1,3))
            # print (chest_start.shape)

          chest_vec = np.cross(hori_axis, vert_axis)
          # print (chest_vec.shape)
          chest_vec = chest_vec.reshape((-1,1,3))
          # print (chest_vec.shape)
          chest_end = chest_vec + chest_start
          # print (chest_end.shape)
          chest_norm_3d = np.concatenate((chest_start, chest_end), axis=1)
          # print (chest_norm_3d.shape)
          # assert False
          
          if 0:
            # Get Rotated chest vector
            chest_end_rot = []
            rot_degree = 90
            rot_rad = np.radians(rot_degree)
            for t in range(chest_norm_3d.shape[0]):
              chest_start_ = chest_start[t].reshape((3,))
              chest_vec_ = chest_vec[t].reshape((3,))
              right_sh_ = right_sh[t]
              left_sh_ = left_sh[t]

              rot_axis_ = right_sh_ - left_sh_
              rot_axis_norm = np.sqrt(np.sum(rot_axis_**2))
              if rot_axis_norm > 0:
                rot_axis_ /= rot_axis_norm
              rot_vec = rot_rad * rot_axis_
              rotation = R.from_rotvec(rot_vec)
              chest_vec_rot_ = rotation.apply(chest_vec_)
              chest_end_rot_ = chest_vec_rot_ + chest_start_
              chest_end_rot.append(chest_end_rot_)
            chest_end_rot = np.array(chest_end_rot).reshape((-1,1,3))
            chest_norm_3d_rot = np.concatenate((chest_start, chest_end_rot), axis=1)
          
          if cfg.viz_chest_3d:
            viz_chest_3d(pose3d_calib, chest_norm_3d, chest_norm_3d_rot, skeleton, dts, pi, cfg)

          ''' f. Project 3D chest normal vector to x-y plane '''
          chest_3d_vec = chest_norm_3d[:,1] - chest_norm_3d[:,0]
          # print (chest_3d_vec.shape)
          chest_2d_vec = chest_3d_vec[:,:2] 
          # chest_2d_vec *= -1
          # print (chest_2d_vec.shape)
          # chest_2d_vec[:,0] *= -1 # left-right flipped
          chest_2d_norm = np.sqrt(np.sum(chest_2d_vec**2, axis=1)).reshape((-1,1))
          # print (chest_2d_norm.shape)
          chest_2d_vec /= chest_2d_norm
          chest_2d_vec *= 15
          # print (chest_2d_vec.shape)
          # assert False
                
          if cfg.viz_chest_2d:
            viz_chest_2d(chest_2d_vec, dts, pi, 'chest_2d', cfg)

          ''' g. Reflect Camera direction in EP6'''
          if 1:
            degree = camera_degree[pi_ip]
          else:
            degree = 360 - camera_degree[pi_ip]
          theta = np.radians(degree)
          c, s = np.cos(theta), np.sin(theta)
          cam_rot = np.array([
            [c, -s],
            [s, c]])
          # ori_cam = cam_rot @ chest_2d_vec.T 
          # ori_cam = ori_cam.T
          ori_cam = chest_2d_vec @ cam_rot

          if cfg.viz_chest_2d_cam:
            viz_chest_2d(ori_cam, dts, pi, 'chest_2d_cam', cfg)
          
          ''' h. Project camera-reflected orientation to EP6 axis '''
          x1, y1 = ori_cam[:,0], -ori_cam[:,1] # vertical exis is fliipped in image axis
          x1 = x1.reshape((-1,1))
          y1 = y1.reshape((-1,1))
          ori_ep6 = np.concatenate([x1, y1], axis=1)

          if cfg.viz_chest_2d_ep6:
            viz_chest_2d(ori_ep6, dts, pi, 'chest_2d_ep6', cfg)
            
          ''' i. Project Foot to EP6 map '''
          avg_feet_positions = extract_avg_feet_positions(kps_seq)
          src = np.array([[y, x, 1] for [x, y, z] in avg_feet_positions], dtype='float32').T
          # print (src.shape)

          # project EP6
          EP6_feet_pos = M.dot(src)
          # print (EP6_feet_pos)
          EP6_feet_pos /= EP6_feet_pos[2,:].reshape((1,-1))
          # print (EP6_feet_pos)
          EP6_feet_pos = EP6_feet_pos[:2].T
          # print (EP6_feet_pos)

          # remove outliers
          EP6_feet_pos[:,0] = np.clip(EP6_feet_pos[:,0], 0, ep6_width-1)
          EP6_feet_pos[:,1] = np.clip(EP6_feet_pos[:,1], 0, ep6_height-1)

          if cfg.viz_occup_ori_ep6_pi:
            viz_occup_ori_ep6_pi(ori_ep6, EP6_feet_pos, ep6_map, dts, pi, cfg)

          ''' collect 1) ori & 2) foot '''
          for i, dt in enumerate(dts):
            # 1) ori
            if dt not in ori_pi:
              ori_pi[dt] = {}
            if pi not in ori_pi[dt]:
              ori_pi[dt][pi] = np.empty((0,2))
            ori = ori_ep6[i].reshape((1,2))
            ori_pi[dt][pi] = np.concatenate((ori_pi[dt][pi], ori))

            # 2) foot
            if dt not in occup_pi:
              occup_pi[dt] = {}
            if pi not in occup_pi[dt]:
              occup_pi[dt][pi] = np.empty((0,2))
            occup = EP6_feet_pos[i].reshape((1,2))
            occup_pi[dt][pi] = np.concatenate((occup_pi[dt][pi], occup))

  if cfg.viz_occup_ori:
    ''' draw occupancy & Orientation observed from all Pis '''
    viz_occup_ori(ori_pi, occup_pi, ep6_map, pi_data, cfg)    

  if cfg.stop_after_viz:
    if cfg.viz_kps_track | cfg.viz_pose3d | cfg.viz_calib_proj | cfg.viz_extrinsic \
      | cfg.viz_extrinsic_smth | cfg.viz_pose3d_calib \
      | cfg.viz_pose3d_calib_match_2d | cfg.viz_chest_3d \
      | cfg.viz_chest_2d | cfg.viz_chest_2d_cam | cfg.viz_chest_2d_ep6 \
      | cfg.viz_occup_ori_ep6_pi:
      assert False

  return occup_pi, ori_pi