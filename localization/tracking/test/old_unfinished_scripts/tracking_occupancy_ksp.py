import pickle as cp

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

output = cp.load(open(file_save, 'rb'))
print ('load from ...', file_save)

assert False, 'tracking with KSP - NO! It crashes due to memory usage. Maybe only small sample is reasonable'
# tracking with KSP - NO! It crashes due to memory usage. Maybe only small sample is reasonable

def convert_v(v,dims,depth):
    # To convert the output of KSP into tracks format
    MAP = np.zeros(dims)
    n_tracks = len(v) // depth
    out = np.zeros((depth,n_tracks,2))
    for t in range(depth-1):
        for i in range(n_tracks):
            out[t,i,0] = v[t*n_tracks + i]/dims[1]
            out[t,i,1] = v[t*n_tracks + i]%dims[1]
            
    return np.int32(out)

import sys
sys.path.append('/Users/hyeokalankwon/Documents/research_local/Emory_local/CEP/EP6/repos/pyKSP/pyKSP')
from ksp import pyKShorthestPathGraph

is_zoom_in = False

depth = 15
file_ep6 = dir_data + '/ep6_map_original.JPG'
ep6_map = plt.imread(file_ep6)
# print (ep6_map.shape)

h, w, c = ep6_map.shape

if is_zoom_in:
  H = 10*h
  W = 10*w
  radius = 250
else:
  H = h
  W = w
  radius = 25

for minute in range(minute_start, minute_end):
  
  q_vector = np.zeros(H*W*depth)
  
  for second in range(depth):
    t = second
    
    occup = output[minute][second]
    # print (occup)
    if is_zoom_in:
      occup *= 10
    # print (occup)
    occup = np.rint(occup).astype(int)
    # print (occup)
    # assert False
    
    # get Q_loc
    Q_loc = np.zeros((H,W)) + 0.00001
    # print (Q_loc.shape)
    # assert False
    
    for i in range(occup.shape[0]):
      p1, p2 = occup[i][0], occup[i][1]
      # print (p1, p2)
      Q_loc[p2, p1] =  0.99      
    # print (Q_loc)
    # assert False
      
    flat_q = np.clip(np.ndarray.flatten(Q_loc),1e-6,0.999999)
    q_vector[H*W*t:H*W*(t+1)] = -np.log(flat_q/(1-flat_q)) # Costs in -log() format
    
  access_points = np.asarray([0]) # Define the access points on your grid
  G = pyKShorthestPathGraph(q_vector,W,H,depth,radius,access_points) #Be
  v = G.getPath(0,depth-1) # From_frame - To_frame (inclusive). Be carreful if you set To_frame>depth - 1, you get a memory leak
  
  print(v)
  del G
  
  out = convert_v(v,(H,W),depth)
  print(out)
      