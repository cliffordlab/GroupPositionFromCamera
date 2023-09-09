import os
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import connection_pose, connection_face
import pickle as cp
import matplotlib.pyplot as plt
import numpy as np

def viz_ori_2d(pi,
              kps,
              face_trans,
              feet_ep6,
              face_ep6,
              ori,
              cfg):

  dir_save = cfg.dir_save + f'/ori_2d'

  for pi in ori_pi:
    dir_save_pi = dir_save + f'/{pi}'
    os.makedirs(dir_save_pi, exist_ok=True)

    for dt in ori_pi[pi]:

      fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,10))

      #---
      ax = axes[0]

      year = dt.year
      month = dt.month
      day = dt.day
      hour = dt.hour
      minute = dt.minute
      second = dt.second

      dir_ymd = cfg.dir_data +f'/{year}/{month:02d}/{day}' 

      dir_hr = dir_ymd + f'/hour_{hour:02d}'
      dir_pi = dir_hr + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
      if os.path.exists(file_frame):
        frame = plt.imread(file_frame)
        # print (frame.shape)
        # assert False
        # print ('load from ...', file_frame)
        frame = np.flip(frame, axis=1)
      else:
        frame = np.ones((frame_height, frame_width, 3))

      ax.imshow(frame)
      for i in range(kps.shape[0]):
        kp = kps[i]
        face_tr = face_trans[i]

        # pose
        for j1, j2 in connection_pose:
          # print (j1, j2)
          # print (kp[j1-1], kp[j2-1])

          y1, x1 = kp[j1-1]
          y2, x2 = kp[j2-1]

          ax.plot(x1, y1, 'o', color='g', markersize=2)
          ax.plot(x2, y2, 'o', color='g', markersize=2)
          ax.plot([x1, x2], [y1, y2], color='g', linewidth=0.5)
        
        # face
        for j1, j2 in connection_face:
          # print (j1, j2)
          # print (face_tr[j1-1], face_tr[j2-1])

          y1, x1 = face_tr[j1-1]
          y2, x2 = face_tr[j2-1]

          ax.plot(x1, y1, 'o', color='r', markersize=2)
          ax.plot(x2, y2, 'o', color='r', markersize=2)
          ax.plot([x1, x2], [y1, y2], color='r', linewidth=0.5)

      #---
      ax = axes[1]

      year = dt.year
      month = dt.month
      day = dt.day
      hour = dt.hour
      minute = dt.minute
      second = dt.second

      ax.imshow(floor)

      # feet
      ax.scatter(feet_ep6[:,0], feet_ep6[:,1], s=10, color='g', marker='o', alpha=0.7)
      
      # face
      for i in range(feet_ep6.shape[0]):
        face_ep6_i = face_ep6[i]

        for j1, j2 in connection_face:
          # print (j1, j2)
          # print (face_tr[j1-1], face_tr[j2-1])

          x1, y1 = face_ep6_i[j1-1]
          x2, y2 = face_ep6_i[j2-1]

          ax.plot(x1, y1, 'o', color='r', markersize=2, alpha=0.7)
          ax.plot(x2, y2, 'o', color='r', markersize=2, alpha=0.7)
          ax.plot([x1, x2], [y1, y2], color='r', linewidth=0.5, alpha=0.7)
      #---
      ax = axes[2]

      ax.imshow(floor)

      for i in range(ori.shape[0]):

        x, y = feet_ep6[i]
        x_nose, y_nose = ori[i]
        # print (x, y)
        # print (x_nose, y_nose)

        dv = np.array([x_nose-x, y_nose-y])
        dv_norm = np.sqrt(np.sum(dv**2))
        if dv_norm > 0:
          dv /= dv_norm
        
        dv *= 17.5
        
        dx, dy = dv[0], dv[1]
        # print (dx, dy)

        ax.arrow(x, y, dx, dy, color='r', width=3)
        # print ('--')

      plt.axis('off')
      plt.tight_layout()
      file_fig = dir_save_pi +f'/{pi}{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}.png'
      plt.savefig(file_fig, bbox_inches='tight', pad_inches=0)
      plt.close()
      print('save in ...', file_fig)

