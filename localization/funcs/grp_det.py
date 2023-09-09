'''
Group detection
Graph-based approach: 
Connected components are groups.
Clque is very conservative approach.
'''
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
import numpy as np
from utils.grouping import detect_direct_interaction
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as cp
import os
from utils.tracking import *
from debug import viz_grp

def grp_det(mv_seq, cfg):
  # Get per frame occupancy from the cleaned trajectory
  occup_track_cleaned = {}
  ori_track_cleaned = {}
  pid_track_cleaned = {}
  for pid in mv_seq:
    dts = mv_seq[pid]['dt']
    locs = mv_seq[pid]['loc']
    oris = mv_seq[pid]['ori']

    for dt, loc, ori in zip(dts, locs, oris):
      if dt not in occup_track_cleaned:
        occup_track_cleaned[dt] = np.empty((0,2))
        ori_track_cleaned[dt] = np.empty((0,2))
        pid_track_cleaned[dt] = []
      
      occup = occup_track_cleaned[dt]
      loc = loc.reshape((1,2))
      occup = np.concatenate((occup, loc))
      occup_track_cleaned[dt] = occup

      orient = ori_track_cleaned[dt]
      ori = ori.reshape((1,2))
      orient = np.concatenate((orient, ori))
      ori_track_cleaned[dt] = orient

      pid_track_cleaned[dt].append(pid)

  # Find one-on-one interaction
  interactions = {}
  for dt in occup_track_cleaned:

    occup = occup_track_cleaned[dt]
    ori = ori_track_cleaned[dt]
    pid = pid_track_cleaned[dt]

    G = nx.Graph()
    for i in range(occup.shape[0]):
      G.add_node(i, loc=occup[i], ori=ori[i], pid=pid[i])
    
    for i in range(occup.shape[0]-1):
      for j in range(i+1, occup.shape[0]):
        loc_i = occup[i]
        loc_j = occup[j]
        dist = np.sqrt(np.sum((loc_i - loc_j)**2))

        ori_i = ori[i]
        ori_j = ori[j]
        facing_inter = detect_direct_interaction(loc_i, ori_i, loc_j, ori_j)

        if (dist < cfg.interaction_th) and facing_inter:
          G.add_edge(i, j)
    
    interactions[dt] = []
    # for cc in nx.enumerate_all_cliques(G):
    for cc in nx.connected_components(G):
      cc = list(cc)
      # print (cc)
      # assert False

      if len(cc) > 1:
        interactions[dt].append(cc)
      
  if cfg.viz_grouping:
    viz_grp(occup_track_cleaned, ori_track_cleaned, interactions, cfg)
  
  if cfg.stop_after_viz:
    if cfg.viz_grouping:
      assert False

  return interactions