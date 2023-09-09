'''
Reflect fixed version of tracker:
tracker_filterpy_ep6_occup_ori_correct_predupdate_fixed.py
'''
import pickle as cp
import matplotlib.pyplot as plt
import cv2
import time
# from tracker_filterpy_ep6_occup_ori import Tracker
# from tracker_filterpy_ep6_occup_ori_correct_predupdate import Tracker
from tracker_filterpy_ep6_occup_ori_correct_predupdate_fixed import Tracker
import numpy as np
from matplotlib import cm

ori_method = '3d'

try:
  dir_data = '/home/hhyeokk/Research/Emory/EP6/exps/est_2022.04.22_data_collection/Localization/Activity area/2022.04.22_16.03_grid_ori/occup'
  file_occup = dir_data + '/multi-view_occup.p'
  occup = cp.load(open(file_occup, 'rb'))
  print ('load from ...', file_occup)
except:
  dir_data = r'C:\Users\hyeok.kwon\Research\Emory\EP6/exps/est_2022.04.22_data_collection/Localization/Activity area/2022.04.22_16.03_grid_ori/occup'
  file_occup = dir_data + '/multi-view_occup.p'
  occup = cp.load(open(file_occup, 'rb'))
  print ('load from ...', file_occup)

file_ori = dir_data + f'/multi-view_ori_{ori_method}.p'
ori = cp.load(open(file_ori, 'rb'))
print ('load from ...', file_ori)

cmap = cm.get_cmap('rainbow')
track_colors = cmap(np.linspace(0, 1, 10))*255

# track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
#                 (255, 255, 0), (127, 127, 255), 
#                 (255, 0, 255), (255, 127, 255),
#         (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]

tracker = Tracker(25, 10)

len_seq = len(occup.keys())
# print (len_seq)
# assert False

data = []
list_dt = []
for i, dt in enumerate(occup):
  occup_dt = occup[dt]
  ori_dt = ori[dt]
  ori_dt_norm = np.sqrt(np.sum(ori_dt**2, axis=1)). reshape((-1,1))
  ori_dt_norm[ori_dt_norm==0] = 1.
  ori_dt /= ori_dt_norm

  centers = np.concatenate([occup_dt, ori_dt], axis=1)
  data.append(centers)
  list_dt.append(dt)

for i in range(len(data)):

  try:
    file_floor = '/home/hhyeokk/Research/Emory/EP6/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    frame = plt.imread(file_floor)
  except:
    file_floor = r'C:\Users\hyeok.kwon\Research\Emory\EP6/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    frame = plt.imread(file_floor)
  
  centers = data[i]
  dt = list_dt[i]
  occup_dt = centers[:,:2]
  ori_dt = centers[:,2:]
  # print (dt, occup_dt.shape, ori_dt.shape, centers.shape)

  if 0:
    # visualize the input
    for j in range(occup_dt.shape[0]):
      occup_x, occup_y = occup_dt[j].astype(int)
      ori_x, ori_y = ori_dt[j]

      tl = (occup_x-10, occup_y-10)
      br = (occup_x+10, occup_y+10)
      
      cv2.rectangle(frame, tl, br, track_colors[j], 1)
      cv2.circle(frame, (occup_x, occup_y), 3, track_colors[j], -1)

      facing_x, facing_y = int(occup_x + ori_x*17.5), int(occup_y + ori_y*17.5)
      cv2.line(frame, (occup_x, occup_y), (facing_x, facing_y), track_colors[j], 2)

    cv2.imshow('image',frame)

    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
  
    continue

  # tracking
  if (len(centers) > 0):
    print ('---')
    print (dt)
    tracker.update(centers)
    print (list(tracker.tracks.keys()))

    for trackId in tracker.tracks:
      j = trackId % 10
      if (len(tracker.tracks[trackId].trace) > 1):
        occup_x = int(tracker.tracks[trackId].trace[-1][0,0])
        occup_y = int(tracker.tracks[trackId].trace[-1][0,1])
        ori_xy = tracker.tracks[trackId].trace[-1][0,2:]
        # print (ori_xy)
        # print (ori_xy.shape)
        # assert False

        ori_xy_norm = np.sqrt(np.sum(ori_xy**2))
        if ori_xy_norm == 0: ori_xy_norm = 1.
        ori_xy /= ori_xy_norm

        ori_x, ori_y = ori_xy
        # print (ori_x, ori_y)

        tl = (occup_x-10, occup_y-10)
        br = (occup_x+10, occup_y+10)

        cv2.rectangle(frame, tl, br, track_colors[j], 1)
        cv2.circle(frame,(occup_x, occup_y), 6, track_colors[j],-1)
        cv2.putText(frame,str(tracker.tracks[trackId].trackId), (occup_x-10,occup_y-20), 0, 0.5, track_colors[j],2)

        facing_x, facing_y = int(occup_x + ori_x*10), int(occup_y + ori_y*10)
        cv2.line(frame, (occup_x, occup_y), (facing_x, facing_y), track_colors[j], 2)
        
        if 1:
          for k in range(len(tracker.tracks[trackId].trace)):
            occup_x = int(tracker.tracks[trackId].trace[k][0,0])
            occup_y = int(tracker.tracks[trackId].trace[k][0,1])
            cv2.circle(frame,(occup_x, occup_y), 3, track_colors[j],-1)

      # detection
      if 1:
        centers = data[i]
        for k in range(centers.shape[0]):
          occup_x, occup_y = centers[k,:2]
          ori_x, ori_y = centers[k,2:] 
          facing_x, facing_y = int(occup_x + ori_x*10), int(occup_y + ori_y*10)
                
          cv2.circle(frame,(int(occup_x),int(occup_y)), 6, (0,0,0),-1)
          # cv2.line(frame, (int(occup_x), int(occup_y)), (facing_x, facing_y), (0,0,0), 2)

    cv2.imshow('image',frame)
    # cv2.imwrite("image"+str(i)+".jpg", frame)
    # images.append(imageio.imread("image"+str(i)+".jpg"))
    
    time.sleep(0.1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
