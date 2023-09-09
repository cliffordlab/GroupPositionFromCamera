from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import linear_sum_assignment

'''
Pi Camera v2 Spec
https://www.raspberrypi.com/documentation/accessories/camera.html

Depth of field: Approx 10 cm to ∞
Focal length: 3.04 mm
Horizontal Field of View (FoV): 62.2 degrees
Vertical Field of View (FoV): 48.8 degrees
'''
dof  = 10 /100
focal_length = 3.04 /10 /100
fovH = 62.2
fovV = 48.8

ceiling_height = 2 # in meters

# camera orientation seeing from south (the entrance gate)
camera_degree = {
  101: 180,
  103: 180,
  106: 180,
  107: 0,
  108: 270,
  114: 225,
  115: 90,
  118: 315,
  120: 180,
  123: 90,
  125: 315,
  129: 270,
  132: 0,
  133: 90,
  135: 135,
  136: 90,
  139: 270,
  140: 315,
  143: 45,
  144: 45,
  145: 270,
  146: 0,
  147: 225,
  148: 180,
  149: 45,
  150: 0,
  151: 180,
  152: 270,
  153: 0,
  154: 90,
  155: 135,
  156: 315,
  157: 90,
  158: 0,
  159: 225,
  160: 225,
  161: 180,
  162: 270,
  163: 45
}

cam_loc = {
  123: [3, 0],
  136: [10, 0],
  146: [12, 2],
  150: [22, 2],
  107: [33, 2],
  151: [0, 8],
  153: [9, 8],
  148: [10, 8],
  106: [21, 8],
  149: [9, 15],
  120: [0, 19],
  155: [0, 21],
  156: [8, 21],
  115: [11, 20],
  158: [22, 20],
  135: [21, 20],
  143: [32, 19],
  154: [5, 23],
  103: [0, 27],
  147: [0, 31],
  152: [5, 29],
  161: [22, 30],
  139: [32, 23],
  125: [29, 26],
  133: [23, 39],
  114: [23, 40],
  101: [0, 40],
  108: [11, 40],
  132: [25, 40],
  118: [32, 40],
  145: [1, 42],
  163: [32, 43],
  144: [11, 49],
  157: [15, 49],
  159: [26, 47],
  162: [4, 51],
  160: [8, 56],
  140: [32, 59],
  129: [21, 59]
}

connection_face = [[1,2], [1,3], [1,4], 
                  [1,5], [2,4], [3,5]]   

connection_pose = [[16, 14], [14, 12], [17, 15], 
            [15, 13], [12, 13], [6, 12], 
            [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], 
            [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]]        

connection = [[16, 14], [14, 12], [17, 15], 
            [15, 13], [12, 13], [6, 12], 
            [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], 
            [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]]        

kps_right_2d = [2, # right eye
		  4, # right ear
		  6, # right shoulder
		  8, # right elbow
		  10, # right wrist
		  12, # right hip
		  14, # right knee
		  16] # right ankle

def timerange(start_time, end_time):
	for n in range(int((end_time - start_time).seconds)+1):
		yield start_time + timedelta(seconds=n)

def kps2dets(kps1):
  x1_y1 = np.amin(kps1, axis=1)
  x2_y2 = np.amax(kps1, axis=1)
  d1 = np.concatenate((x1_y1, x2_y2), axis=1)
  return d1  

def kps_matching_between_frames(kps1, kps2, verbose=False):
  # hungarian matching

  d1 = kps2dets(kps1)
  d2 = kps2dets(kps2)

  cost_mat = np.zeros((kps1.shape[0], kps2.shape[0]))
  for i in range(d1.shape[0]):
    for j in range(d2.shape[0]):

      iou_score = iou(d1[i], d2[j])
      cost_mat[i, j] = -iou_score
  
  row_ind, col_ind = linear_sum_assignment(cost_mat)
  if verbose:
    print (row_ind)
    print (col_ind)

  # remove invalid matchings
  val_ind = np.ones(row_ind.shape, dtype=bool)
  for i, (r, c) in enumerate(zip(row_ind, col_ind)):
    iou_score = cost_mat[r,c]
    if verbose:
      print (r, c, iou_score)
    if abs(iou_score) < 1e-6:
      val_ind[i] = False
  row_ind = row_ind[val_ind]
  col_ind = col_ind[val_ind]
  if verbose:
    print (row_ind)
    print (col_ind)

  return row_ind, col_ind

def occup_matching_between_frames(occup1, occup2, th=17.5*2): # 1m: 17.5 px
  # hungarian matching
  cost_mat = np.zeros((occup1.shape[0], occup2.shape[0]))
  for i in range(occup1.shape[0]):
    for j in range(occup2.shape[0]):

      d1 = occup1[i]
      d2 = occup2[j]

      l2_dist = np.sqrt(np.sum((d1-d2)**2))
      cost_mat[i,j] = l2_dist

  row_ind, col_ind = linear_sum_assignment(cost_mat)

  # remove invalid matchings
  if th is None:
    return row_ind, col_ind
    
  val_ind = np.ones(row_ind.shape, dtype=bool)
  for i, (r, c) in enumerate(zip(row_ind, col_ind)):
    l2_dist = cost_mat[r,c]
    if l2_dist > th:
      val_ind[i] = False
  row_ind = row_ind[val_ind]
  col_ind = col_ind[val_ind]

  return row_ind, col_ind

def iou(bb_test, bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
      + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)

  return(o)

class hungarian_kps():
  def __init__(self, distance='bbox', th=1e-6, max_pid=None):
    assert distance in ['bbox', 'kps']
    if th is not None:
      assert distance == 'bbox', "'distance' variable needs to be bbox when 'th' is provided"

    self.distance = distance

    self.th = th

    self.assign = None
    self.dets_prev = None
    self.max_pid = max_pid
  
  def update(self, kps):
    '''
    input: kps (N, 17, 2)

    outputs: id
    '''
    if self.distance == 'bbox':
      dets = kps2dets(kps)
    else:
      dets = kps

    if self.assign is None:
      self.assign = np.arange(dets.shape[0])
      self.dets_prev = dets
      if self.max_pid is None:
        self.max_pid = np.amax(self.assign)
      else:
        self.max_pid += np.amax(self.assign)
      return self.assign
    
    cost_mat = np.zeros((self.assign.shape[0], dets.shape[0]))
    # cost_mat[:] = 1e6
    # print (cost_mat)
    # print (cost_mat.shape)
    # print (self.dets_prev.shape, dets.shape)
    for i1, d1 in enumerate(self.dets_prev):
      for i2, d2 in enumerate(dets):
        if self.distance == 'bbox':
          iou_score = iou(d1[:4], d2[:4])
          cost_mat[i1,i2] = -iou_score
        elif self.distance == 'kps':
          l2_dist = np.sqrt(np.sum((d1-d2)**2))
          cost_mat[i1,i2] = l2_dist
        else:
          raise ValueError(self.distance)
    # print (cost_mat)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    # print (row_ind)
    # print (col_ind)
    # assert False

    assign = np.empty((dets.shape[0],))
    assign[:] = np.nan
    for r, c in zip(row_ind, col_ind):
      if self.th is not None \
      and abs(cost_mat[r,c]) < self.th:
        self.max_pid += 1
        assign[c] = self.max_pid
      else:
        assign[c] = self.assign[r]
    # print (self.assign)
    # print (assign)
    # assert False

    for i, ass in enumerate(assign):
      if np.isnan(ass):
        self.max_pid += 1
        assign[i] = self.max_pid
    # print (assign)
    # print ('----------')
    
    self.assign = assign
    self.dets_prev = dets
    return self.assign

class hungarian_occup():
  def __init__(self, max_pid=None, th=np.nan):

    self.th = th

    self.assign = None
    self.dets_prev = None
    self.max_pid = max_pid
  
  def update(self, dets):
    '''
    dets: [[x1, y1]]
    outputs: id
    '''
    # print (dets.shape)

    if self.assign is None:
      self.assign = np.arange(dets.shape[0])
      self.dets_prev = dets
      if self.max_pid is None:
        self.max_pid = np.amax(self.assign)
      else:
        self.max_pid += np.amax(self.assign)
      return self.assign
    
    cost_mat = np.zeros((self.assign.shape[0], dets.shape[0]))
    # cost_mat[:] = 1e6
    # print (cost_mat)
    # print (cost_mat.shape)
    # print (self.dets_prev.shape, dets.shape)
    for i1, d1 in enumerate(self.dets_prev):
      for i2, d2 in enumerate(dets):
        # print (i1, d1)
        # print (i2, d2)
        # assert False
        l2_dist = np.sqrt(np.sum((d1-d2)**2))
        cost_mat[i1,i2] = l2_dist
    # print (cost_mat)
    # assert False

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    # print (row_ind)
    # print (col_ind)
    # assert False

    # for r, c in zip(row_ind, col_ind):
    # 	print (r, c, cost_mat[r,c])
    # assert False

    assign = np.empty((dets.shape[0],))
    assign[:] = np.nan
    for r, c in zip(row_ind, col_ind):
      if self.th is not None \
      and cost_mat[r,c] > self.th:
        self.max_pid += 1
        assign[c] = self.max_pid
      else:
        assign[c] = self.assign[r]
    # print (self.assign)
    # print (assign)
    # assert False

    for i, ass in enumerate(assign):
      if np.isnan(ass):
        self.max_pid += 1
        assign[i] = self.max_pid
    # print (assign)
    # print ('----------')
    
    self.assign = assign
    self.dets_prev = dets
    return self.assign

#-------------------------------

# Get multi-view occupancy
def extract_avg_feet_positions(keypoints):
    """
    This function extracts the average feet positions of all people from keypoints stored by posenet.
    It returns the average of position of left and right feet for each individual.
    
    Inputs:
        keypoints - 10x17x2 matrix containing the x,y coordinates of all keypoints 17 for maximum 10 people in the frame.
                    This matrix corresponds to one frame.
        
    Returns:
        avg_feet_positions - 10x2 array of x,y coordinates of average of left and right foot positions for maximum
                              of 10 people in the frame.
    """
    
    footX = []
    footY = []
    avg_feet_positions = []
    
    for i in range(len(keypoints)):
        foot_X_temp = [] # Stores avg X positions of feet for all people in one frame 
        foot_Y_temp = [] # Stores avg Y positions of feet for all people in one frame 
        foot_temp = [] # Stores avg x,y positions of feet for all people in one frame 
        
        for j in range(len(keypoints[i])):
            if (j+1)%17 == 0:
                foot_right_X = keypoints[i][j][0]
                foot_right_Y = keypoints[i][j][1]
                foot_left_X = keypoints[i][j-1][0]
                foot_left_Y = keypoints[i][j-1][1]
                foot_X_temp.append(int((foot_right_X + foot_left_X)/2))
                foot_Y_temp.append(int((foot_right_Y + foot_left_Y)/2))
                foot_temp.append((int((foot_right_X + foot_left_X)/2), int((foot_right_Y + foot_left_Y)/2), 1))
        footX.append(foot_X_temp)
        footY.append(foot_Y_temp)
        avg_feet_positions.append(np.array(foot_temp))
    
    avg_feet_positions = np.array(avg_feet_positions)
    avg_feet_positions = avg_feet_positions.reshape((len(keypoints),3))
    
    return avg_feet_positions