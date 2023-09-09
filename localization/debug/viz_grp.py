import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def viz_grp_t(occup, orient, grps, dt, dir_copy_group, cfg):
  year = dt.year
  month = dt.month
  day = dt.day
  hour = dt.hour
  minute = dt.minute
  second = dt.second

  file_copy_occup_pi = dir_copy_group + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_occup_pi):
    fig, ax = plt.subplots()
    ax.imshow(ep6_map)
    ax.scatter(occup[:,0], occup[:,1], s=10, color='r', marker='^', alpha=0.7) 

    # orientation
    for j in range(len(occup)):
      occ = locs[j]
      ori = orient[j]
      x1, y1 = ori
      x2, y2 = occ
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='k')      

    for grp in grps:
      # print (grp)
      for i in range(len(grp)-1):
        idx1 = grp[i]
        idx2 = grp[i+1]
        # print (idx1, idx2)
        loc1 = occup[idx1]
        loc2 = occup[idx2]

        x = [loc1[0], loc2[0]]
        y = [loc1[1], loc2[1]]
        # print (x, y)
    
        ax.plot(x, y, c='k')
        
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)

def viz_grp(occup_track_cleaned, ori_track_cleaned, interactions, cfg):

  dir_copy_group = cfg.dir_save + f'/group_th_{cfg.interaction_th}'
  if cfg.use_ep6_grid:
    dir_copy_group += '_grid'
  dir_copy_group += '_ori'
  os.makedirs(dir_copy_group, exist_ok=True)

  for dt in occup_track_cleaned:
    occup = occup_track_cleaned[dt]
    orient = ori_track_cleaned[dt]
    grps = interactions[dt]

    viz_grp_t(occup, orient, grps, dt, dir_copy_group, cfg)
