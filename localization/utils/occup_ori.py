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
#from mpl_toolkits.mplot3d import Axes3D
#from utils.occup_ori import *
#import pickle as cp
from debug import viz_kps_track, viz_pose3d, viz_calib_proj, viz_extrinsic, viz_pose3d_calib_match_2d, viz_chest_3d, viz_chest_2d, viz_occup_ori_ep6_pi, viz_occup_ori, viz_occup_ep6_pi, viz_occup
from scipy.spatial.transform import Rotation as R

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

def collect_ori(ori_ep6_pi_pid, pi_pid_seq, cfg):
  
  ori_pi = {}
  for pi in ori_ep6_pi_pid:
    for pid in ori_ep6_pi_pid[pi]:
      ori_ep6 = ori_ep6_pi_pid[pi][pid]
      dts = pi_pid_seq[pi][pid]['dt']

      for i, dt in enumerate(dts):
        # orientation
        if dt not in ori_pi:
          ori_pi[dt] = {}
        if pi not in ori_pi[dt]:
          ori_pi[dt][pi] = np.empty((0,2))
        ori = ori_ep6[i].reshape((1,2))
        ori_pi[dt][pi] = np.concatenate((ori_pi[dt][pi], ori))
  
  return ori_pi

def collect_occup(EP6_feet_pos_pi_pid, pi_pid_seq, pi_data, cfg):
  
  occup_pi = {}  
  for pi in EP6_feet_pos_pi_pid:
    for pid in EP6_feet_pos_pi_pid[pi]:
      EP6_feet_pos = EP6_feet_pos_pi_pid[pi][pid]
      dts = pi_pid_seq[pi][pid]['dt']

      for i, dt in enumerate(dts):
        # foot
        if dt not in occup_pi:
          occup_pi[dt] = {}
        if pi not in occup_pi[dt]:
          occup_pi[dt][pi] = np.empty((0,2))
        occup = EP6_feet_pos[i].reshape((1,2))
        occup_pi[dt][pi] = np.concatenate((occup_pi[dt][pi], occup))

  if cfg.viz_occup:
    ''' draw occupancy & Orientation observed from all Pis '''
    viz_occup(occup_pi, pi_data, cfg)   
    assert False

  return occup_pi

def proj_foot_ep6(kps_seq_pi_pid_np, pi_pid_seq, cfg):

  file_ep6 = cfg.dir_data +'/ep6_floorplan_measured_half_gridded_1_meter.jpg'
  ep6_map = plt.imread(file_ep6)
  ep6_height, ep6_width, _ = ep6_map.shape

  EP6_feet_pos_pi_pid = {}
  for pi in kps_seq_pi_pid_np:

    pi_ip = int(pi[2:5])
    dir_proj_mat = cfg.dir_exp + '/proj_mat'
    file_save = dir_proj_mat +f'/pi_ip_{pi_ip}.npy'
    assert os.path.exists(file_save), pi_ip
    if not os.path.exists(file_save):
      print (file_save, '... not exist')
      continue
    M = np.load(file_save)
    
    EP6_feet_pos_pi_pid[pi] = {}
    for pid in kps_seq_pi_pid_np[pi]:
      kps_seq = kps_seq_pi_pid_np[pi][pid]
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

      EP6_feet_pos_pi_pid[pi][pid] = EP6_feet_pos
  
  if cfg.viz_occup_ep6_pi:
    for pi in EP6_feet_pos_pi_pid:
      for pid in EP6_feet_pos_pi_pid[pi]:
        EP6_feet_pos = EP6_feet_pos_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_occup_ep6_pi(EP6_feet_pos, dts, pi, pid, cfg)
    # assert False
    
  return EP6_feet_pos_pi_pid

def get_ori_ep6(ori_cam_pi_pid, pi_pid_seq, cfg):
  ori_ep6_pi_pid = {}
  for pi in ori_cam_pi_pid:
    ori_ep6_pi_pid[pi] = {}
    for pid in ori_cam_pi_pid[pi]:
      ori_cam = ori_cam_pi_pid[pi][pid]
      x1, y1 = ori_cam[:,0], -ori_cam[:,1] # vertical exis is fliipped in image axis
      x1 = x1.reshape((-1,1))
      y1 = y1.reshape((-1,1))
      ori_ep6 = np.concatenate([x1, y1], axis=1)

      ori_ep6_pi_pid[pi][pid] = ori_ep6
  
  if cfg.viz_chest_2d_ep6:
    for pi in ori_ep6_pi_pid:
      for pid in ori_ep6_pi_pid[pi]:
        ori_ep6 = ori_ep6_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_chest_2d(ori_ep6, dts, pi, pid, 'chest_2d_ep6', cfg)
        
  return ori_ep6_pi_pid

def get_ori_cam(chest_2d_vec_pi_pid, pi_pid_seq, cfg):
  ori_cam_pi_pid = {}
  for pi in chest_2d_vec_pi_pid:
    ori_cam_pi_pid[pi] = {}
    pi_ip = int(pi[2:5])
    if 1:
      degree = camera_degree[pi_ip]
    else:
      degree = 360 - camera_degree[pi_ip]
    for pid in chest_2d_vec_pi_pid[pi]:
      chest_2d_vec = chest_2d_vec_pi_pid[pi][pid]

      theta = np.radians(degree)
      c, s = np.cos(theta), np.sin(theta)
      cam_rot = np.array([
        [c, -s],
        [s, c]])
      # ori_cam = cam_rot @ chest_2d_vec.T 
      # ori_cam = ori_cam.T
      ori_cam = chest_2d_vec @ cam_rot

      ori_cam_pi_pid[pi][pid] = ori_cam
  
  if cfg.viz_chest_2d_cam:
    for pi in ori_cam_pi_pid:
      for pid in ori_cam_pi_pid[pi]:
        ori_cam = ori_cam_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_chest_2d(ori_cam, dts, pi, pid, 'chest_2d_cam', cfg)

  return ori_cam_pi_pid

def proj_chest_norm_3d_xy(chest_norm_3d_pi_pid, pi_pid_seq, cfg):
  chest_2d_vec_pi_pid = {} 
  for pi in chest_norm_3d_pi_pid:
    chest_2d_vec_pi_pid[pi] = {}
    for pid in chest_norm_3d_pi_pid[pi]:
      chest_norm_3d = chest_norm_3d_pi_pid[pi][pid]
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
      chest_2d_vec_pi_pid[pi][pid] = chest_2d_vec

  if cfg.viz_chest_2d:
    for pi in chest_2d_vec_pi_pid:
      for pid in chest_2d_vec_pi_pid[pi]:
        chest_2d_vec = chest_2d_vec_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_chest_2d(chest_2d_vec, dts, pi, pid, 'chest_2d', cfg)

  return chest_2d_vec_pi_pid

def get_chest_norm_3d(pose3d_calib_pi_pid, pi_pid_seq, cfg):
  chest_norm_3d_pi_pid = {}
  chest_start_pi_pid = {}
  chest_vec_pi_pid = {}
  left_sh_pi_pid = {}
  right_sh_pi_pid = {}
  for pi in pose3d_calib_pi_pid:
    chest_norm_3d_pi_pid[pi] = {}
    chest_start_pi_pid[pi] = {}
    chest_vec_pi_pid[pi] = {}
    left_sh_pi_pid[pi] = {}
    right_sh_pi_pid[pi] = {}
    for pid in pose3d_calib_pi_pid[pi]:
      pose3d_calib = pose3d_calib_pi_pid[pi][pid]
  
      left_sh = pose3d_calib[:, 11]
      right_sh = pose3d_calib[:, 14]

      left_sh_pi_pid[pi][pid] = left_sh
      right_sh_pi_pid[pi][pid] = right_sh

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

      chest_start_pi_pid[pi][pid] = chest_start
      chest_vec_pi_pid[pi][pid] = chest_vec
      chest_norm_3d_pi_pid[pi][pid] = chest_norm_3d
      
  if cfg.viz_chest_3d:
    for pi in pose3d_calib_pi_pid:
      for pid in pose3d_calib_pi_pid[pi]:
        pose3d_calib = pose3d_calib_pi_pid[pi][pid]
        chest_norm_3d = chest_norm_3d_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_chest_3d(pose3d_calib, chest_norm_3d, dts, pi, 'chest_3d', cfg)
  
  if 0:
    # Get Rotated chest vector
    # chest 3d rotate for sholder axis
    chest_norm_3d_rot_pi_pid = {}
    for pi in pose3d_calib_pi_pid:
      chest_norm_3d_rot_pi_pid[pi] = {}
      for pid in pose3d_calib_pi_pid[pi]:
        chest_norm_3d = chest_norm_3d_pi_pid[pi][pid]
        chest_start = chest_start_pi_pid[pi][pid]
        chest_vec = chest_vec_pi_pid[pi][pid]

        left_sh = left_sh_pi_pid[pi][pid]
        right_sh = right_sh_pi_pid[pi][pid]
    
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

        chest_norm_3d_rot_pi_pid[pi][pid] = chest_norm_3d

    if cfg.viz_chest_3d_rot:
      for pi in pose3d_pi_pid:
        for pid in pose3d_pi_pid[pi]:
          pose3d_calib = pose3d_calib_pi_pid[pi][pid]
          chest_norm_3d_rot = chest_norm_3d_rot_pi_pid[pi][pid]
          dts = pi_pid_seq[pi][pid]['dt']
          viz_chest_3d(pose3d_calib, chest_norm_3d_rot, dts, pi, 'chest_3d_rot', cfg)
          
  return chest_norm_3d_pi_pid

def calib_pose3d(pose3d_pi_pid, calib_3d_pi_pid, pi_pid_seq, cfg):
  pose3d_calib_pi_pid = {}
  for pi in pose3d_pi_pid:
    pose3d_calib_pi_pid[pi] = {}
    for pid in pose3d_pi_pid[pi]:
      prediction = pose3d_pi_pid[pi][pid]
      calib_3d = calib_3d_pi_pid[pi][pid]
  
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
      pose3d_calib_pi_pid[pi][pid] = pose3d_calib

  if cfg.viz_pose3d_calib_match_2d:
    for pi in pose3d_pi_pid:
      for pid in pose3d_pi_pid[pi]:
        pose3d_calib = pose3d_calib_pi_pid[pi][pid]
        kps_seq = kps_seq_pi_pid_np[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_pose3d_calib_match_2d(pose3d_calib, kps_seq, dts, pi, pid, cfg)
  
  return pose3d_calib_pi_pid

def smth_calib(calib_3d_pi_pid, pi_pid_seq, cfg):
  for pi in calib_3d_pi_pid:
    for pid in calib_3d_pi_pid[pi]:

      # 1) smooth rotation
      proj_rot = calib_3d_pi_pid[pi][pid]['rot']
      proj_rot = proj_rot.reshape((proj_rot.shape[0],3))
      rot = R.from_rotvec(proj_rot)
      quat = rot.as_quat()
      for ch in range(quat.shape[1]):
        quat_ch = quat[:,ch]
        quat_ch = Series(quat_ch).rolling(cfg.smooth_calb_len, min_periods=1, center=True).mean().to_numpy()
        quat[:,ch] = quat_ch
      rot = R.from_quat(quat)
      proj_rot = rot.as_rotvec()
      
      calib_3d_pi_pid[pi][pid]['rot'] = proj_rot.reshape((proj_rot.shape[0],3,1))
      
      # smoothe trans
      proj_trans = calib_3d_pi_pid[pi][pid]['trans']
      proj_trans = proj_trans.reshape((proj_trans.shape[0],3))
      for ch in range(proj_trans.shape[1]):
        trans_ch = proj_trans[:,ch]
        trans_ch = Series(trans_ch).rolling(cfg.smooth_calb_len, min_periods=1, center=True).mean().to_numpy()
        proj_trans[:,ch] = trans_ch
      
      calib_3d_pi_pid[pi][pid]['trans'] = proj_trans.reshape((proj_trans.shape[0],3,1))

  if cfg.viz_extrinsic_smth:
    for pi in calib_3d_pi_pid:
      for pid in calib_3d_pi_pid[pi]:
        calib_3d = calib_3d_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        dt = dts[0]
        viz_extrinsic(calib_3d, dt, pi, pid, True, cfg)

  return calib_3d_pi_pid

def pnp_2d_3d(kps_seq_pi_pid_np, pose3d_pi_pid, pi_pid_seq, cfg):

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

  solvepnp_flag = cv2.SOLVEPNP_EPNP # So this gives the best improvment
  
  calib_3d_pi_pid = {}
  for pi in kps_seq_pi_pid_np:
    calib_3d_pi_pid[pi] = {}
    for pid in kps_seq_pi_pid_np[pi]:

      dir_save_extrinsic = cfg.dir_save + f'/extrinsics/{pi}/{int(pid):03d}'
      os.makedirs(dir_save_extrinsic, exist_ok=True)
      file_rot = dir_save_extrinsic + '/rot.npy'
      file_trans = dir_save_extrinsic + '/trans.npy'

      if cfg.use_interm_ori3d_pnp \
      and os.path.exists(file_rot) \
      and os.path.exists(file_trans):
        rot_arr = np.load(file_rot)
        trans_arr = np.load(file_trans)
      else:
        kps_seq = kps_seq_pi_pid_np[pi][pid]
        prediction = pose3d_pi_pid[pi][pid]
        rot_arr = []
        trans_arr = []
        for t in range(kps_seq.shape[0]):
          try:
            pose2d = kps_seq[t]
            pose3d = prediction[t]
          except:
            print(kps_seq.shape, prediction.shape)
            assert False
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

          rot_arr.append(rotation_vector)
          trans_arr.append(translation_vector)

        rot_arr = np.array(rot_arr)
        trans_arr = np.array(trans_arr)
        
        np.save(file_rot, rot_arr)
        np.save(file_trans, trans_arr)

      calib_3d_pi_pid[pi][pid] = {'rot': rot_arr,
                                  'trans': trans_arr}
        
  if cfg.viz_calib_proj:
    for pi in kps_seq_pi_pid_np:
      for pid in kps_seq_pi_pid_np[pi]:
        calib_3d = calib_3d_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        kps_seq = kps_seq_pi_pid_np[pi][pid]
        prediction = pose3d_pi_pid[pi][pid]
        viz_calib_proj(kps_seq, prediction, dts, calib_3d, cfg)

  if cfg.viz_extrinsic:
    for pi in calib_3d_pi_pid:
      for pid in calib_3d_pi_pid[pi]:
        calib_3d = calib_3d_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        dt = dts[0]
        viz_extrinsic(calib_3d, dt, pi, pid, False, cfg)

  return calib_3d_pi_pid

def lift_2d_to_3d(kps_seq_pi_pid_np, cfg):

  # prepare 3D lifting model
  sys.path.append(cfg.path_videopose3d)

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

  pose3d_pi_pid = {}
  for pi in kps_seq_pi_pid_np:
    pose3d_pi_pid[pi] = {}
    for pid in kps_seq_pi_pid_np[pi]:
      kps_seq = kps_seq_pi_pid_np[pi][pid]

      dir_save_pose3d = cfg.dir_save + f'/pose3d/{pi}/{int(pid):03d}'
      os.makedirs(dir_save_pose3d, exist_ok=True)

      file_pose3d = dir_save_pose3d + '/pred_pose3d.npy'

      if cfg.use_interm_pose3d and os.path.exists(file_pose3d):
        prediction = np.load(file_pose3d)
      else:
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
        np.save(file_pose3d, prediction)

      pose3d_pi_pid[pi][pid] = prediction

  if cfg.viz_pose3d:
    for pi in pose3d_pi_pid:
      for pid in pose3d_pi_pid[pi]:
        prediction = pose3d_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_pose3d(prediction, skeleton, dts, pi, pid, cfg)

  return pose3d_pi_pid

def collect_kps_np(pi_data, pi_pid_seq, cfg):

  kps_seq_pi_pid_np = {}
  for pi in pi_pid_seq:      
    kps_seq_pi_pid_np[pi] = {}
    for pid in pi_pid_seq[pi]:
      dts = pi_pid_seq[pi][pid]['dt']
      rids = pi_pid_seq[pi][pid]['rid']
      if len(dts)> 1:

        # collect sequence
        kps_seq = np.empty((0, 17, 2))
        for dt, rid in zip(dts, rids):
          kps = pi_data[pi][dt][rid].reshape((1, 17, 2))
          kps_seq = np.concatenate((kps_seq, kps))
        kps_seq_pi_pid_np[pi][pid] = kps_seq

  if cfg.viz_kps_track:
    for pi in kps_seq_pi_pid_np:
      for pid in kps_seq_pi_pid_np[pi]:
        dts = pi_pid_seq[pi][pid]['dt']
        kps_seq = kps_seq_pi_pid_np[pi][pid]
        viz_kps_track(kps_seq, dts, pi, pid, cfg)

  return kps_seq_pi_pid_np

def track_pose2d_pi(pi_data, cfg):

  pi_pid_seq = {}
  for pi in pi_data:
    pi_pid_seq[pi] = {}

    pi_ip = int(pi[2:5])
    dir_proj_mat = cfg.dir_exp + '/proj_mat'
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
          if pid not in pi_pid_seq[pi]:
            pi_pid_seq[pi][pid] = {'dt': [], 'rid': []}
          if max_pid is None:
            max_pid = pid
          elif max_pid < pid:
            max_pid = pid
            
          pi_pid_seq[pi][pid]['dt'].append(dt_curr)
          pi_pid_seq[pi][pid]['rid'].append(rid)

  return pi_pid_seq