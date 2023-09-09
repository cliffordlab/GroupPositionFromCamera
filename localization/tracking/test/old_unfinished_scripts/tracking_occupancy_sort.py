import os
import pickle as cp
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from matplotlib import cm

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  -> test x gt
  my version:
  det x track
  """
  dist_mat = np.empty((bb_test.shape[0], bb_gt.shape[0]))
  for m in range(bb_test.shape[0]):
    for n in range(bb_gt.shape[0]):
      loc_m = bb_test[m][:2]
      loc_n = bb_gt[n][:2]
      # print (loc_m, loc_n)
      # assert False
      dist = np.sqrt(np.sum((loc_m - loc_n)**2))
      dist_mat[m,n] = dist
      
  return dist_mat

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  return bbox.reshape((2,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  # print (x.shape)
  # assert False
  return x[:2].reshape((1,2))

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2) 
    self.kf.F = np.array([[1,0,0.1,0],
                          [0,1,0,0.1],
                          [0,0,1,0],
                          [0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],
                          [0,1,0,0]])

    self.kf.R *= 0.01
    self.kf.P *= 0.1 
    self.kf.Q[2:,2:] *= 0.01
  
    self.kf.x[:2] = convert_bbox_to_z(bbox)    
    # print (self.kf.x.shape)
    # print (convert_bbox_to_z(bbox).shape)
    # assert False
    
    # print ('kf.R')
    # print (self.kf.R)
    # print (self.kf.R.shape)
    # print ('kf.P')
    # print (self.kf.P)
    # print (self.kf.P.shape)
    # print ('kf.Q')
    # print (self.kf.Q)
    # print (self.kf.Q.shape)
    # assert False
    
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if((self.kf.x[6]+self.kf.x[2])<=0):
    #   self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)
  
def associate_detections_to_trackers(detections,
                                     trackers,
                                     iou_threshold = 25):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,3),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix < iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]] > iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=25):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 3))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 3))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    # print (trks)
    # assert False
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
    # print (matched)
    # print (unmatched_dets)
    # print (unmatched_trks)
    # assert False

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        # print (trk.get_state().shape)
        # print (d.shape)
        # assert False
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,3))
  
  
from pprint import pprint

dir_root = '/Users/hyeokalankwon/Research/Emory_local/CEP/EP6'

year = '2021'
month = '12'
day = '01'
# hour = '12'
hour = '11'
minute_start = 50 # 1
minute_end = 51 # 60
minute = 50

dir_exp = dir_root + '/exps/occupancy'
dir_save = dir_exp + f'/{year}.{month}.{day}_{hour}.{minute}'
file_save = dir_save + '/multi-view.p'

occupancy = cp.load(open(file_save, 'rb'))
print ('load from ...', file_save)

occupancy = occupancy[minute]

track_ids = {}
track_times = {}

mot_tracker = Sort() #create instance of the SORT tracker  
for second in range(0, 60):
  occup = occupancy[second]
  # print (occup)
  # print (occup.shape)
  # assert False
  
  trackers = mot_tracker.update(occup)
  
  for d in trackers:
    # print (d)
    # print (d.shape)
    # print(f'{second:d}, {int(d[2]):d}, {d[0]:.2f}, {d[1]:.2f}')
    
    pos = d[:2]
    tid = d[2]
    
    if tid not in track_ids:
      track_ids[tid] = {'second': [], 'pos': []}
    track_ids[tid]['second'].append(second)
    track_ids[tid]['pos'].append(pos)
    
    if second not in track_times:
      track_times[second] = {}    
    if tid not in track_times[second]:
      track_times[second][tid] = pos

pprint (track_ids)
# assert False

# visualize
dir_data = dir_root + '/data'
file_ep6 = dir_data + '/ep6_map_original.JPG'
ep6_map = plt.imread(file_ep6)

list_tid = list(track_ids.keys())
# print (list_tid)
cmap = cm.get_cmap('rainbow')
c_list = cmap(np.linspace(0, 1, len(list_tid)))

dir_exp = dir_root + '/exps/occupancy_tracking'

# all track in minute.
dir_save = dir_exp + f'/{year}.{month}.{day}_{hour}.{minute}'
os.makedirs(dir_save, exist_ok=True)

fig, ax = plt.subplots()
ax.imshow(ep6_map)
for i, tid in enumerate(list_tid):
  
  locs = np.array(track_ids[tid]['pos'])
  for j in range(len(locs)-1):
    ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=c_list[i])
  
file_save = dir_save + f'/all_track.png'
plt.savefig(file_save)
plt.close()  
print ('save in ...', file_save)


if 0:
  # each second
  dir_save = dir_exp + f'/{year}.{month}.{day}_{hour}.{minute}/seconds'
  os.makedirs(dir_save, exist_ok=True)

  for second in range(1, 60):
    file_save = dir_save + f'/{int(second):03d}.png'
    
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    track_curr = track_times[second]
    track_prev = track_times[second-1]
    for tid in track_curr:
      pos_curr = track_curr[tid]
      if tid not in track_prev:
        continue
      pos_prev = track_prev[tid]
      
      xs = [pos_prev[0], pos_curr[0]]
      ys = [pos_prev[1], pos_curr[1]]
      i_tid = list_tid.index(tid)
      ax.plot(xs, ys, c=c_list[i_tid], label=tid)
    
    plt.legend(loc='upper right')
    plt.savefig(file_save)
    plt.close()
    print ('save in ...', file_save)
    
if 0:
  # per id
  dir_save = dir_exp + f'/{year}.{month}.{day}_{hour}.{minute}/ids'
  os.makedirs(dir_save, exist_ok=True)

  for i, tid in enumerate(list_tid):
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    
    locs = np.array(track_ids[tid]['pos'])
    for j in range(len(locs)-1):
      ax.plot(locs[j:j+2,0],  locs[j:j+2,1], c=c_list[i])
    
    file_save = dir_save + f'/id_{int(tid):03d}.png'
    plt.savefig(file_save)
    plt.close()  
    print ('save in ...', file_save)
  