import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import cv2

connection = [[16, 14], [14, 12], [17, 15], 
            [15, 13], [12, 13], [6, 12], 
            [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], 
            [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]]        

invTrans = transforms.Compose([ 
              transforms.Normalize(
                  mean = [ 0., 0., 0. ],
                  std = [ 1/0.229, 1/0.224, 1/0.225 ]),
              transforms.Normalize(
                  mean = [ -0.485, -0.456, -0.406 ],
                  std = [ 1., 1., 1. ]),
                              ])

def viz_mebow_img(input, degree, 
                  file_frame, kps, bbox,
                  pi, pid, ts, cfg):

  input = input.reshape(3, 256, 192)
  input = invTrans(input)
  input = np.transpose(input, (1,2,0))

  dir_save = cfg.dir_save + f'/ori/{pi}_{pid}'
  os.makedirs(dir_save, exist_ok=True)

  year = ts.year
  month = ts.month
  day = ts.day
  hour = ts.hour
  minute = ts.minute
  second = ts.second

  file_save = dir_save + f'/{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

  fig, axes = plt.subplots(nrows=2)

  if isinstance(file_frame, str):
    frame = plt.imread(file_frame)
    frame = cv2.flip(frame, 1)
  else:
    frame = file_frame.copy() # data loaded

  # bbox
  x, y, w, h = bbox.astype(int)
  start_point = (x, y)
  end_point = (x+w, y+h)
  thickness = 2
  color = (255, 0, 0)
  frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
  axes[0].imshow(frame)

  # kps
  color = 'b'
  for j1, j2 in connection:
    # print (j1, j2)
    # print (kp[j1-1], kp[j2-1])

    y1, x1 = kps[j1-1]
    y2, x2 = kps[j2-1]

    axes[0].plot(x1, y1, 'o', color=color, markersize=2)
    axes[0].plot(x2, y2, 'o', color=color, markersize=2)
    axes[0].plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
  axes[0].axis('off')

  # input to the model
  axes[1].imshow(input)
  axes[1].set_title(f'degree: {degree}')
  plt.axis('off')
  plt.tight_layout()
  plt.savefig(file_save, bbox_inches='tight',pad_inches = 0)
  plt.close()
  print ('save in ...', file_save)
  # assert False 
