import pickle as cp
from pprint import pprint
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction/tracking/kf')
from kf_pose import Tracker
from matplotlib import cm
import numpy as np
import os
from datetime import datetime, timedelta
import cv2
import matplotlib.pyplot as plt
import random

frame_height = 481
frame_width = 641

connection = [[16, 14], [14, 12], [17, 15], 
            [15, 13], [12, 13], [6, 12], 
            [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], 
            [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]]        

n_color = 20
cmap = cm.get_cmap('rainbow')
track_colors = cmap(np.linspace(0, 1, n_color))*255

def timerange(start_time, end_time):
	for n in range(int((end_time - start_time).seconds)+1):
		yield start_time + timedelta(seconds=n)

dir_data = "/home/hhyeokk/Research/EP6/exps/est_2022.04.22_data_collection/Localization/Activity area/kps_smth"
file_pi_data = dir_data + '/pi_data.p'
pi_data = cp.load(open(file_pi_data, 'rb'))
print ('load from ...', file_pi_data)

dir_save = '/home/hhyeokk/Research/EP6/exps/tracking_kf'

# load kps dataset
# pprint(list(pi_data.keys()))
# ['pi106.pi.bmi.emory.edu',
#  'pi123.pi.bmi.emory.edu',
#  'pi139.pi.bmi.emory.edu',
#  'pi148.pi.bmi.emory.edu',
#  'pi157.pi.bmi.emory.edu',
#  'pi154.pi.bmi.emory.edu',
#  'pi153.pi.bmi.emory.edu',
#  'pi107.pi.bmi.emory.edu',
#  'pi151.pi.bmi.emory.edu',
#  'pi136.pi.bmi.emory.edu']

no_trace = True

pi = 'pi106.pi.bmi.emory.edu'
dir_save_pi = dir_save +f'/{pi}'
if no_trace:
  dir_save_pi += '_no_trace'
os.makedirs(dir_save_pi, exist_ok=True)

# pprint(list(pi_data[pi]))
list_t = list(pi_data[pi])
t_min, t_max = np.amin(list_t), np.amax(list_t) 

tracker = Tracker(100, 30)

i_frame = 0
for t in timerange(t_min, t_max):

  skip_det = False
  if random.random() <= 0.5:
    skip_det = True

  if not skip_det and t in pi_data[pi]:
    kps_det = pi_data[pi][t]
    # print (kps_det.shape)
    # assert False
    kps_det = kps_det.reshape((-1, 34))
    tracker.update(kps_det)
  else:
    tracker.update()
    print ('no detection! only predict ...')
    # assert False

  year = t.year
  month = t.month
  day = t.day
  hour = t.hour
  minute = t.minute
  second = t.second

  dir_ymd = '/home/hhyeokk/Research/EP6/data' + f'/{year}/{month:02d}/{day}'
  dir_hr = dir_ymd + f'/hour_{hour:02d}'
  dir_pi = dir_hr + f'/{pi}'
  dir_vid = dir_pi + '/videos'
  file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
  if os.path.exists(file_frame):
    frame = plt.imread(file_frame)
    # print (frame.shape)
    # assert False
    # print ('load from ...', file_frame)
    frame = np.flip(frame, axis=1).astype(np.uint8)
  else:
    frame = 255*np.ones((frame_height, frame_width, 3)).astype(np.uint8)
  
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_pred = frame.copy()
  
  # print (len(tracker.tracks))
  list_trackId = list(tracker.tracks.keys())
  print (list_trackId)

  n_pred = 0
  for trackId in tracker.tracks:
    j = trackId % n_color
    # print (len(tracker.tracks[trackId].trace))
    # assert False

    if (len(tracker.tracks[trackId].trace) > 0):
      n_pred += 1
      # print (trackId)
      kps_pred = tracker.tracks[trackId].trace[-1][0].astype(int)
      # if not skip_det:
      #   print (trackId)
      #   print (kps_pred)
      # print (kps_pred.shape)
      # assert False
      kps_pred = kps_pred.reshape((17,2))

      for j1, j2 in connection:
        # print (kps_pred[j1-1])
        # print (kps_pred[j2-1])
        x1, y1 = kps_pred[j1-1]
        x2, y2 = kps_pred[j2-1]

        cv2.line(frame_pred, (y1, x1), (y2, x2), track_colors[j], 2)
      
      x1, y1 = kps_pred[0]+10 # nose
      cv2.putText(frame_pred, str(trackId), (y1, x1), 0, 0.5, track_colors[j],2)
      
      # trace
      if not no_trace:
        for k in range(len(tracker.tracks[trackId].trace)-1):
          kps_pred = tracker.tracks[trackId].trace[k][0].astype(int)
          kps_pred = kps_pred.reshape((17,2))
          for j1, j2 in connection:
            x1, y1 = kps_pred[j1-1]
            x2, y2 = kps_pred[j2-1]
            cv2.line(frame_pred, (y1, x1), (y2, x2), track_colors[j], 2)
  
  print (f'{n_pred}/{len(tracker.tracks)}')

  # detection
  if not skip_det:
    frame_det = frame.copy()
    kps_det = kps_det.astype(int)
    kps_det = kps_det.reshape((-1,17,2))
    for i in range(kps_det.shape[0]):
      for j1, j2 in connection:
        x1, y1 = kps_det[i,j1-1]
        x2, y2 = kps_det[i,j2-1]      
        cv2.line(frame_det, (y1, x1), (y2, x2), (0,0,0), 2)
  else:
    frame_det = frame
  frame = cv2.addWeighted(frame_det, 0.5, frame_pred, 0.5, 0)

  # cv2.imshow('image',frame)
  # time.sleep(0.1)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #   cv2.destroyAllWindows()
  #   break

  if skip_det:
    file_save = dir_save_pi + f'/{i_frame:03d}_{pi}{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}_skip_det.png'
  else:
    file_save = dir_save_pi + f'/{i_frame:03d}_{pi}{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}.png'

  cv2.imwrite(file_save, frame)
  print ('save in ...', file_save)
  # assert False
  i_frame += 1
  print ('---')
