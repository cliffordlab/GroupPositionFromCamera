''' 
Multi-view: Person matching and association across camera 
+ weighted integration considering camera distance to the object

Current version:

Multi-view Localization:
- Only use occupation
TODO: will update if the orientation looks okay.
dist = L_2(occup_i, occup_j) + lambda * -cosine_similarity

Multi-view Orientation:
Since multi-view localization does not use orientation yet,
Just get the average vector of connected components.
TODO: Once multi-view localization is updated with orientation, then only take average of those are connected for this case.
'''

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import os 
import pickle as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from utils.tracking import *
from debug import viz_occup_mv_pi, viz_occup_mv

xs = np.arange(77, 1129, 17.5)
ys = np.arange(90, 669, 17.5)

def occup_ori_mv(occup_pi, ori_pi, pi_data, cfg):
  list_pi = list(pi_data.keys())
  list_pi.sort()

  occup_mv = {}
  ori_mv = {}
  for dt in occup_pi:
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    timestep = f'{year}.{month:02}.{day:02}_{hour:02}.{minute:02}.{second:02d}'

    G = nx.Graph()
    for pi in occup_pi[dt]:
      occup = occup_pi[dt][pi]
      for i in range(occup.shape[0]):
        loc_i = occup[i]
        node = f'{pi}_{i}'
        if node not in G:
          G.add_node(node, loc=loc_i)    
    
    for i in range(len(list_pi)-1):
      for j in range(i+1, len(list_pi)):
        pi_i = list_pi[i]
        pi_j = list_pi[j]

        if pi_i not in occup_pi[dt] \
        or pi_j not in occup_pi[dt]:
          continue

        occup_i = occup_pi[dt][pi_i]
        occup_j = occup_pi[dt][pi_j]

        dist_mat = np.empty((occup_i.shape[0], occup_j.shape[0]))
        dist_mat[:] = np.inf
        for r in range(occup_i.shape[0]):
          for c in range(occup_j.shape[0]):
            loc_m = occup_i[r]
            loc_n = occup_j[c]

            node1 = f'{pi_i}_{r}'
            if node1 not in G:
              G.add_node(node1, loc=loc_m)

            node2 = f'{pi_j}_{c}'
            if node2 not in G:
              G.add_node(node2, loc=loc_n)

            dist = np.sqrt(np.sum((loc_m - loc_n)**2))
            dist_mat[r,c] = dist

        row_ind, col_ind = linear_sum_assignment(dist_mat)

        for r,c in zip(row_ind, col_ind):
          dist = dist_mat[r, c]
        
          if dist < cfg.overlap_th:
            node1 = f'{pi_i}_{r}'
            node2 = f'{pi_j}_{c}'
            # print (node1, node2)
            G.add_edge(node1, node2)
          
    if 0:
      # find connected components
      for cc in nx.connected_components(G):
        # check if connected component include multiple occups from a pi
        # this should not happen
        # if so, then remove edge
        if 1:
          list_cc_pi = []
          for pi_o in cc:
            pi = pi_o.split('_')[0]
            if pi not in list_cc_pi:
              list_cc_pi.append(pi)
            else:
              print (cc)
              assert False
        print (cc, len(cc))
      print ('---------------------')   

    occup_mv[dt] = []
    ori_mv[dt] = []
    for cc in nx.connected_components(G):
      cc = list(cc)
      n_node = len(cc)

      if n_node == 1:
        ''' single node CC '''
        pi, i = cc[0].split('_')
        i = int(i)
        occup = occup_pi[dt][pi][i]
        # print (occup.shape)
        
        ori = ori_pi[dt][pi][i]
        # print (ori.shape)
        # assert False

      elif cfg.mv_camdist:
        ''' Consider camera distance '''
        # print (cc)
        # print (n_node)
        # assert False

        occup = np.empty((n_node,2))
        ori = np.empty((n_node,2))
        weight = np.empty((n_node,1))

        for i_node, pi_o in enumerate(cc):
          pi, i = pi_o.split('_')
          i = int(i)

          pi_id = int(pi[2:5])
          cam_loc_pi = cam_loc[pi_id]
          cam_loc_pi_x = xs[int(cam_loc_pi[1])]
          cam_loc_pi_y = ys[int(cam_loc_pi[0])]
          cam_loc_pi = np.array([cam_loc_pi_x, cam_loc_pi_y])
          # print (pi_id, cam_loc_pi)

          _occup = occup_pi[dt][pi][i]          
          # print (occup.shape)
          # assert False
          occup[i_node] = _occup
          # print (_occup)

          dist = np.sqrt(np.sum((cam_loc_pi-_occup)**2))/17.5
          weight[i_node] = 1/dist

          # get orientation vectors as average of connected components
          _ori = ori_pi[dt][pi][i]
          ori[i_node] = _ori
        
        # print (weight)
        weight /= np.sum(weight)
        # print (weight)
        # print (weight.shape)

        occup = weight.T @ occup
        # print (occup.shape)
        # assert False
        # occup = occup.reshape((2,))

        ori = weight.T @ ori
        # print (ori.shape)
        # assert False
        # ori = ori.reshape((2,))
        # print ('----')
      else:
        ''' naive averaging ''' 
        occup = np.zeros((2,))
        ori = np.zeros((2,))
        for pi_o in cc:
          pi, i = pi_o.split('_')
          i = int(i)
          _occup = occup_pi[dt][pi][i]
          # print (occup.shape)
          # assert False
          occup += _occup

          # get orientation vectors as average of connected components
          _ori = ori_pi[dt][pi][i]
          ori += _ori

        occup /= n_node
        ori /= n_node

      occup = occup.reshape((2,))
      ori = ori.reshape((2,))

      try:
        assert occup.shape[0] == 2, (occup, occup.shape)
      except:
        print (occup)
        print (n_node)
        print (dt, pi, i,)
        assert False
      try:
        assert ori.shape[0] == 2, (ori, ori.shape)
      except:
        print (ori)
        print (dt, pi, i)
        assert False

      occup_mv[dt].append(occup)

      ori_norm = np.sqrt(np.sum(ori**2))
      if ori_norm > 0:
        ori /= ori_norm

      if 0:
        # separate start & end points
        _ori = np.zeros((2,2))
        _ori[0,0] = occup # ori start
        _ori[0,1] = occup + ori

        ori_mv[dt].append(_ori)
      else:
        # only the vector
        ori_mv[dt].append(ori)
      # assert False, 'Need to figure this out.'
      # print (ori.shape)
      # assert False

    # print (occup_mv[dt])
    # print (ori_mv[dt])
    occup_mv[dt] = np.array(occup_mv[dt])
    ori_mv[dt] = np.array(ori_mv[dt])
    # print (occup_mv[dt].shape)
    # print (ori_mv[dt].shape)
    # assert False

    if cfg.viz_occup_mv_pi:
      ''' draw connected components from occupancy observed from all Pis '''
      ori_pi_dt = ori_pi[dt]
      viz_occup_mv_pi(G, pi_data, ori_pi_dt, dt, ep6_map, cfg)

  # assert False

  if cfg.viz_occup_mv:
    viz_occup_mv(occup_mv, ori_mv, cfg)
    
  ''' Save Multi-view occupancy  '''
  # pushti's code
  # dir_occup = dir_save + '/occupancy'
  # dir_save = dir_occup + f'/{year}.{month}.{day}_{hour}.{minute}'
  # dir_save = '/Users/jigardesai/Downloads/test'

  # hyeok's code
  # dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
  # if cfg.use_ep6_grid:
  #   dir_copy += '_grid'
  # dir_copy += '_ori'
  # dir_save_occup = dir_copy + '/occup'

  # os.makedirs(dir_save_occup, exist_ok=True)

  file_save = cfg.dir_save + '/multi-view_occup.p'
  cp.dump(occup_mv, open(file_save, 'wb'))
  print ('save in ...', file_save)

  file_save = cfg.dir_save + f'/multi-view_ori.p'
  cp.dump(ori_mv, open(file_save, 'wb'))
  print ('save in ...', file_save)

  if cfg.stop_after_viz:
    if cfg.viz_occup_mv | cfg.viz_occup_mv_pi:
      assert False

  return occup_mv, ori_mv