'''
Reflect fixed version of tracker that has options to collect state variables:
tracker_filterpy_ep6_occup_ori_correct_predupdate_fixed_optionCollect.py
'''
import pickle as cp
import matplotlib.pyplot as plt
import cv2
import time
# from tracker_filterpy_ep6_occup_ori import Tracker
# from tracker_filterpy_ep6_occup_ori_correct_predupdate import Tracker
# from tracker_filterpy_ep6_occup_ori_correct_predupdate_fixed import Tracker
# from tracker_filterpy_ep6_occup_ori_correct_predupdate_fixed_optionCollect import Tracker
from tracker_filterpy_ep6_occup_ori_correct_predupdate_fixed_optionCollect_smooth import Tracker
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

collect_states=True
max_trace_length=None

tracker = Tracker(25, 10,
                  collect_states=collect_states, 
                  max_trace_length=max_trace_length)

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

dir_save = r'C:\Users\hyeok.kwon\Research\Emory\EP6/exps/tracking_kf/test'

if 0:
  # draw detections
  try:
    file_floor = '/home/hhyeokk/Research/Emory/EP6/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    frame = plt.imread(file_floor)
  except:
    file_floor = r'C:\Users\hyeok.kwon\Research\Emory\EP6/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    frame = plt.imread(file_floor)
    
  for i in range(len(data)):
    centers = data[i]
    occup_dt = centers[:,:2]
    ori_dt = centers[:,2:]

    for j in range(occup_dt.shape[0]):
      occup_x, occup_y = occup_dt[j].astype(int)
      ori_x, ori_y = ori_dt[j]

      cv2.circle(frame, (occup_x, occup_y), 3, (255, 0, 0), -1)

      facing_x, facing_y = int(occup_x + ori_x*17.5), int(occup_y + ori_y*17.5)
      cv2.line(frame, (occup_x, occup_y), (facing_x, facing_y), (255, 0, 0), 2)

  file_save = dir_save + '/detections.png'
  cv2.imwrite(file_save, frame)
  print ('save in ...', file_save)
  # assert False

for i in range(len(data)):
  centers = data[i]
  dt = list_dt[i]
  occup_dt = centers[:,:2]
  ori_dt = centers[:,2:]

  tracker.update(centers, dt)
  print (list(tracker.tracks.keys()))
  print ('---')

tracker.close()

print (list(tracker.tracks_collected.keys()))
# assert False

for trackId in tracker.tracks_collected:
  print (f'trackId: {trackId}')
  j = trackId % 10

  try:
    file_floor = '/home/hhyeokk/Research/Emory/EP6/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    frame = plt.imread(file_floor)
  except:
    file_floor = r'C:\Users\hyeok.kwon\Research\Emory\EP6/data/ep6_floorplan_measured_half_gridded_1_meter.jpg'
    frame = plt.imread(file_floor)

  ts = tracker.tracks_collected[trackId].ts
  # print (ts)
  len_ts = len(ts)
  num_sec = (np.amax(ts) - np.amin(ts)).seconds + 1
  print (len_ts, num_sec)
  assert len_ts == num_sec
  
  means = tracker.tracks_collected[trackId].means
  covariance = tracker.tracks_collected[trackId].covariance
  means = np.array(means)
  covariance = np.array(covariance)
  print (means.shape)
  print (covariance.shape)

  frame_filter = frame.copy()
  for t in range(means.shape[0]):
    occup_x = int(means[t][0])
    occup_y = int(means[t][1])
    cv2.circle(frame_filter,(occup_x, occup_y), 3, (0, 0, 0),-1)

  means_smoothed = tracker.tracks_collected[trackId].means_smoothed
  covariance_smoothe = tracker.tracks_collected[trackId].covariance_smoothe
  means_smoothed = np.array(means_smoothed)
  covariance_smoothe = np.array(covariance_smoothe)

  frame_smooth = frame.copy()
  for t in range(means.shape[0]):
    occup_x = int(means_smoothed[t][0])
    occup_y = int(means_smoothed[t][1])
    cv2.circle(frame_smooth,(occup_x, occup_y), 3, track_colors[j],-1)
  
  print (means_smoothed.shape)
  print (covariance_smoothe.shape)

  frame = cv2.addWeighted(frame_filter, 0.5, frame_smooth, 0.5, 0)
  cv2.imshow('image',frame)
  # assert False

  file_save = dir_save + f'/{trackId:03d}_smoothed.png'
  cv2.imwrite(file_save, frame)
