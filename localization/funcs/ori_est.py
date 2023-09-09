import sys
sys.path.append('/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/social_interaction')
from utils.occup_ori import *
from utils.tracking import *
from debug import viz_occup_ori_ep6_pi, viz_ori_2d, viz_chest_2d
from .ori_est_mebow import ori_est_mebow
from .ori_est_mebow_img_preload import ori_est_mebow_img
# from .ori_est_mebow_img import ori_est_mebow_img
from transform import Transforms, Plotting, Utils
import os

def ori_mebow_est(pi_pid_seq, kps_seq_pi_pid_np, 
            EP6_feet_pos_pi_pid, cfg):

  if cfg.ori_method == 'mebow':

    # chest normal vector on x-y plane from camera viewpoint
    print ('-- MEBOW: chest normal vector on x-y plane from camera viewpoint ')
    if cfg.mebow_image:
      # assert False
      chest_2d_vec_pi_pid, chest_2d_degree_pi_pid = ori_est_mebow_img(kps_seq_pi_pid_np, pi_pid_seq, cfg)
    else:
      chest_2d_vec_pi_pid, chest_2d_degree_pi_pid = ori_est_mebow(kps_seq_pi_pid_np)

    # file_save = cfg.dir_save + '/chest_2d_degree.p'
    # cp.dump(chest_2d_degree_pi_pid, open(file_save, 'wb'))
    # print ('save in ...', file_save)

    if cfg.viz_chest_2d:
      for pi in chest_2d_vec_pi_pid:
        for pid in chest_2d_vec_pi_pid[pi]:
          chest_2d_vec = chest_2d_vec_pi_pid[pi][pid]
          dts = pi_pid_seq[pi][pid]['dt']
          viz_chest_2d(chest_2d_vec, dts, pi, pid, 'chest_2d', cfg)
      assert False

    ''' g. Reflect Camera direction in EP6 '''
    print ('-- Reflect Camera direction in EP6 ')
    ori_cam_pi_pid = get_ori_cam(chest_2d_vec_pi_pid, pi_pid_seq, cfg)

    ''' h. Project camera-reflected orientation to EP6 axis '''
    print ('-- Project camera-reflected orientation to EP6 axis ')
    ori_ep6_pi_pid = get_ori_ep6(ori_cam_pi_pid, pi_pid_seq, cfg)
  
  elif cfg.ori_method == '2d':
    print ('-- 2D-based Orientation estimation ')
    
    ori_ep6_pi_pid = {}
    for pi in kps_seq_pi_pid_np:
      pi_ip = int(pi[2:5])
      dir_proj_mat = cfg.dir_exp + '/proj_mat'
      file_save = dir_proj_mat +f'/pi_ip_{pi_ip}.npy'
      assert os.path.exists(file_save), pi_ip
      if not os.path.exists(file_save):
        print (file_save, '... not exist')
        continue
      M = np.load(file_save)
        
      ori_ep6_pi_pid[pi] = {}
      for pid in kps_seq_pi_pid_np[pi]:
        kps = kps_seq_pi_pid_np[pi][pid]
        # print (kps.shape)
        # assert False
        # print ('--')

        '''
        translate facial keypoints centered around feet,
        considering nose to be at the feet.
        '''

        face = kps[:,:5]
        # print (face.shape)
        nose = kps[:,0]
        # print (nose.shape)
        # print ('--')

        feet = kps[:,15:]
        # print (feet)
        # print (feet.shape)
        feet = np.mean(feet, axis=1)
        # print (feet)
        # print (feet.shape)
        # assert False
        # print ('--')

        diff = feet - nose
        # print (diff.shape)
        diff = diff.reshape((-1, 1, 2))
        # print (diff.shape)
        face_trans = face + diff
        # print (face_trans.shape)
        # assert False
          
        '''
        project translated feet and face keypoints on ep6
        '''

        # feet
        src = np.array([[y, x, 1] for [x, y] in feet], dtype='float32').T
        # print (src.shape)

        feet_ep6 = M.dot(src)
        # print (feet_ep6.shape)
        feet_ep6 /= feet_ep6[2,:].reshape((1,-1))
        # print (feet_ep6.shape)
        feet_ep6 = feet_ep6[:2].T
        # print (feet_ep6.shape)
        # assert False

        # face
        face_ep6 = np.zeros(face_trans.shape)
        # print (face_ep6.shape)
        for i in range(face_trans.shape[0]):
          face_trans_i = face_trans[i]
          # print (face_trans_i.shape)
 
          src = np.array([[y, x, 1] for [x, y] in face_trans_i], dtype='float32').T
          # print (src.shape)

          face_ep6_i = M.dot(src)
          # print (face_ep6_i.shape)
          face_ep6_i /= face_ep6_i[2,:].reshape((1,-1))
          # print (face_ep6_i.shape)
          face_ep6_i = face_ep6_i[:2].T
          # print (face_ep6_i.shape)

          face_ep6[i] = face_ep6_i
          # assert False

        '''
        Get orientation.
        Assumption: The average of eye & ear locations are the facing direction ...
        I don't believe this at all...
        '''
        # print (face_ep6.shape)
        pointing = np.mean(face_ep6[:,1:], axis=1).reshape((-1, 1, 2))
        nose = face_ep6[:,0].reshape((-1, 1, 2))
        # print (pointing.shape)
        # print (nose.shape)

        ori = pointing - nose
        ori_norm = np.sqrt(np.sum(ori**2))
        if ori_norm > 0:
          ori /= ori_norm
        # print (ori.shape)

        ori_ep6_pi_pid[pi][pid] = ori
        # assert False
    
        if cfg.viz_ori_2d:
          viz_ori_2d(pi, kps, face_trans,
              feet_ep6, face_ep6, ori, cfg)

  elif cfg.ori_method == '3d':

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

    ''' a. lift 2D -> 3D '''
    print ('-- lift 2D -> 3D')
    pose3d_pi_pid = lift_2d_to_3d(kps_seq_pi_pid_np, cfg)

    ''' b. PnP 2D & 3d '''
    print ('-- PnP 2D & 3d')
    calib_3d_pi_pid = pnp_2d_3d(kps_seq_pi_pid_np, pose3d_pi_pid, pi_pid_seq, cfg)

    ''' c. smooth calibrated 1) rotation & 2) translation ''' 
    print ('-- smooth calibrated 1) rotation & 2) translation')
    calib_3d_pi_pid = smth_calib(calib_3d_pi_pid, pi_pid_seq, cfg)

    ''' d. calibrate 3d pose 
    The calibrated 3D pose represents the 3D pose view from camera.
    So, the chest vector should be okay to reflect where the person is looking at.
    The noise will be smoothed temporally.
    '''
    print ('-- calibrate 3d pose ')
    pose3d_calib_pi_pid = calib_pose3d(pose3d_pi_pid, calib_3d_pi_pid, pi_pid_seq, cfg)

    ''' e. Get 3D chest normal vector '''
    print ('-- Get 3D chest normal vector ')
    chest_norm_3d_pi_pid = get_chest_norm_3d(pose3d_calib_pi_pid, pi_pid_seq, cfg)

    ''' f. Project 3D chest normal vector to x-y plane '''
    print ('-- Project 3D chest normal vector to x-y plane ')
    chest_2d_vec_pi_pid = proj_chest_norm_3d_xy(chest_norm_3d_pi_pid, pi_pid_seq, cfg)

    ''' g. Reflect Camera direction in EP6 '''
    print ('-- Reflect Camera direction in EP6 ')
    ori_cam_pi_pid = get_ori_cam(chest_2d_vec_pi_pid, pi_pid_seq, cfg)

    ''' h. Project camera-reflected orientation to EP6 axis '''
    print ('-- Project camera-reflected orientation to EP6 axis ')
    ori_ep6_pi_pid = get_ori_ep6(ori_cam_pi_pid, pi_pid_seq, cfg)
          
  else:
    assert False, cfg.ori_method

  ''' collect orientation '''
  print ('-- collect orientation ')
  ori_pi = collect_ori(ori_ep6_pi_pid, pi_pid_seq, cfg)

  if cfg.viz_occup_ori_ep6_pi:
    for pi in EP6_feet_pos_pi_pid:
      for pid in EP6_feet_pos_pi_pid[pi]:
        EP6_feet_pos = EP6_feet_pos_pi_pid[pi][pid]
        ori_ep6 = ori_ep6_pi_pid[pi][pid]
        dts = pi_pid_seq[pi][pid]['dt']
        viz_occup_ori_ep6_pi(ori_ep6, EP6_feet_pos, dts, pi, pid, cfg)
    assert False
  
  return ori_pi

def ori_2d_est(pi_pid_seq, kps_seq_pi_pid_np, cfg):
    
    file_ep6 = cfg.dir_data +'/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    ep6_map = plt.imread(file_ep6)
    ep6_height, ep6_width, _ = ep6_map.shape
    
    T = Transforms()
    U = Utils()
    
    ori_2d_pi = {}
    for pi in kps_seq_pi_pid_np:
    
      pi_ip = int(pi[2:5])
      dir_proj_mat = cfg.dir_exp + '/proj_mat'
      file_save = dir_proj_mat +f'/pi_ip_{pi_ip}.npy'
      assert os.path.exists(file_save), pi_ip
      if not os.path.exists(file_save):
        print (file_save, '... not exist')
        continue
      M = np.load(file_save)
      
      ori_2d_pi[pi] = {}
      for pid in kps_seq_pi_pid_np[pi]:
          
        kps_seq = kps_seq_pi_pid_np[pi][pid]
        
        translated_face = T.translate(kps_seq)
        keypoints_transformed = np.zeros((np.shape(translated_face)[0], np.shape(translated_face)[1], 3))
        for i in range(np.shape(keypoints_transformed)[1]):
            src = np.array([[y, x, 1] for [x, y] in translated_face[:,i,:]], dtype='float32').T
    
            # project EP6
            kpts_pos = M.dot(src)
            # print (EP6_feet_pos)
            kpts_pos /= kpts_pos[2,:].reshape((1,-1))
            # print (EP6_feet_pos)
            kpts_pos = kpts_pos[:2].T
            # print (EP6_feet_pos)
    
            # remove outliers
            kpts_pos[:,0] = np.clip(kpts_pos[:,0], 0, ep6_width-1)
            kpts_pos[:,1] = np.clip(kpts_pos[:,1], 0, ep6_height-1)
            kpts_pos = np.array([np.hstack((k,1)) for k in kpts_pos])
            
            keypoints_transformed[:,i,:] = kpts_pos
        
        orientation_vectors = T.face_to_nose_vector(keypoints_transformed) 
        orientation_degrees = U.convert_angle_to_degrees(orientation_vectors)
        ori_2d_pi[pi][pid] = orientation_vectors
        
    # Restructure ori_2d_pi to look like ori_mebow_pi
    ori_2d_restructured = collect_ori_2d(ori_2d_pi, pi_pid_seq, cfg)
        
    return ori_2d_restructured


def collect_ori_2d(ori_2d_ep6_pi_pid, pi_pid_seq, cfg):
  ori_pi = {}
  for pi in ori_2d_ep6_pi_pid:
    for pid in ori_2d_ep6_pi_pid[pi]:
      ori_ep6 = ori_2d_ep6_pi_pid[pi][pid]
      dts = pi_pid_seq[pi][pid]['dt']

      for i, dt in enumerate(dts):
        # orientation
        if dt not in ori_pi:
          ori_pi[dt] = {}
        if pi not in ori_pi[dt]:
          ori_pi[dt][pi] = np.empty((0,4))
        ori = ori_ep6[i].reshape((1,4))
        ori_pi[dt][pi] = np.concatenate((ori_pi[dt][pi], ori))

  return ori_pi