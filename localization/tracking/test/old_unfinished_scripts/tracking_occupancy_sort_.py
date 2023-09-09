''' Code exploration. not the final code '''

import pickle as cp

import numpy as np
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as loc.
  """
  count = 0
  def __init__(self,loc):
    # print (loc.shape)
    # assert False
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2) 
    self.kf.F = np.array([[1,0,1,0],
                          [0,1,0,1],
                          [0,0,1,0],
                          [0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],
                          [0,1,0,0]])

    self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[2:,2:] *= 0.01

    self.kf.x[:2] = loc.reshape((2,1))
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,loc):
    """
    Updates the state vector with observed loc.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(loc)

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x[:2].reshape((1,2)))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    # print (self.kf.x.shape)
    # print (self.kf.x)
    # print (self.kf.x[:2])
    # assert False
    return self.kf.x[:2].reshape((1,2))
  
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
      loc_m = bb_test[m]
      loc_n = bb_gt[n]
      # print (loc_m, loc_n)
      # assert False
      dist = np.sqrt(np.sum((loc_m - loc_n)**2))
      dist_mat[m,n] = dist
      
  return dist_mat
      
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

  def update(self, dets=np.empty((0, 2))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 2))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1]]
      if np.any(np.isnan(pos)):
        to_del.append(t)
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
        # print (d)
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

track_result = {}

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
    print(f'{second:d}, {int(d[2]):d}, {d[0]:.2f}, {d[1]:.2f}')
  