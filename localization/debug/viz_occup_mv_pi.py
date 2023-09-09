import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def viz_occup_mv_pi(G, pi_data, ori_pi_dt, dt, ep6_map, cfg):
  ''' draw connected components from occupancy observed from all Pis '''

  year = dt.year
  month = dt.month
  day = dt.day
  hour = dt.hour
  minute = dt.minute
  second = dt.second
  
  list_pi = list(pi_data.keys())
  list_pi.sort()
  c_pis = cfg.cmap(np.linspace(0, 1, len(list_pi)))
  pi_color = {}
  for pi, c in zip(list_pi, c_pis):
    pi_color[pi] = c

  if 0:
    dir_copy = cfg.dir_save + f'/{year}.{month:02d}.{day:02d}_{hour:02d}.{minute:02d}'
    if cfg.use_ep6_grid:
      dir_copy += '_grid'
    dir_copy += '_ori'
    dir_copy_occup = dir_copy + f'/occup_connected_component_ov_{cfg.overlap_th}'
  else:
    dir_copy_occup = cfg.dir_save + f'/occup_connected_component_ov_{cfg.overlap_th}'
  os.makedirs(dir_copy_occup, exist_ok=True)

  file_copy_occup_pi = dir_copy_occup + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  if not os.path.exists(file_copy_occup_pi):
    # pprint (list(G.nodes(data=True)))
    # print (len(G.nodes(data=True)))
    # for node in G.nodes(data=True):
    #   print (node)
    #   print (node[0])
    #   print (node[1])
    #   print (node[1]['loc'])
    #   print ('--------')
    # print ('========================')
    # pprint (list(G.edges(data=True)))
    # print (len(G.edges(data=True)))
    # assert False

    fig, ax = plt.subplots()
    ax.imshow(ep6_map)

    for node in G.nodes(data=True):
      pi, i = node[0].split('_')
      pi_ip = pi[2:5]
      loc = node[1]['loc']
      color = pi_color[pi]
      ax.scatter(loc[0], loc[1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

      # draw orientation
      _ori = ori_pi_dt[pi][i]
      x1, y1 = ori
      x2, y2 = loc
      # ori_end = ori + occup
      x3, y3 = x2 + x1, y2 + y1
      ax.plot([x2, x3], [y2, y3], color='b')
    # for pi in occup_pi[dt]:
    #   occup = occup_pi[dt][pi]
    #   color = pi_color[pi]
    #   pi_ip = pi[2:5]
    #   ax.scatter(occup[:,0], occup[:,1], s=10, color=color, marker='^', alpha=0.7, label=pi_ip)

    if len(G.edges(data=True)) > 0:    
      # pprint (list(G.nodes(data=True)))
      # print (len(G.edges(data=True)))
      # print (len(G.edges()))
      # print (list(G.edges(data=True)))
      # print (list(G.edges()))
      # print ('=========')
      # for edge in G.edges():
      #   print (edge)
      #   node1, node2 = edge
      #   print (node1)
      #   print (G.nodes[node1])
      #   print (G.nodes[node1]['loc'])
      #   print (node2)
      #   print (G.nodes[node2])
      #   print (G.nodes[node2]['loc'])
      #   print ('-----')
      # assert False

      for edge in G.edges():
        n1, n2 = edge
        loc_m = G.nodes[n1]['loc']
        loc_n = G.nodes[n2]['loc']

        x = [loc_m[0], loc_n[0]]
        y = [loc_m[1], loc_n[1]]
    
        ax.plot(x, y, c='k')

    ax.legend()

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_copy_occup_pi, bbox_inches='tight',pad_inches = 0)
    plt.close()
    print ('save in ...', file_copy_occup_pi)
