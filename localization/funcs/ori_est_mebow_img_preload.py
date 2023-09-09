'''
Image-based MEBOW + Preload inputs
'''

from argparse import Namespace
import sys
sys.path.append('/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/social_interaction')
#sys.path.append('/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/MEBOW')
sys.path.append('/opt/scratchspace/chegde/EP6/repo/MEBOW')
from lib.utils.transforms import get_affine_transform
from lib import models
from lib.config import cfg, update_config
import cv2
from pandas import Series
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import os
from mebow.dataset import COCO_HOE_Dataset_Img
from debug import viz_mebow_img
from pprint import pprint

def ori_est_mebow_img(kps_seq_pi_pid_np, pi_pid_seq, _cfg):

  '''
  - Preload input bbox images
  - Run inference in batch
  '''
  
  #sys.path.append(_cfg.path_mebow_repo)
  
  #-------------------
  #from lib import Models
  #from lib.Config import cfg, update_config
  #-------------------

  args = Namespace(cfg=_cfg.path_mebow_cfg, dataDir='', logDir='', modelDir='', opts=[], prevModelDir='')

  update_config(cfg, args)

  cudnn.benchmark = cfg.CUDNN.BENCHMARK
  torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
  torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

  model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
      cfg, is_train=False)
  
  print (f'=> loading model from {cfg.TEST.MODEL_FILE}')
  model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)

  model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

  model.eval()

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  )

  dataloader = COCO_HOE_Dataset_Img(cfg,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  normalize,]),
                              flip_hori=True)

  # rearrange to pi -> dt -> pid, kps
  kps_pi_dt = {}
  for pi in kps_seq_pi_pid_np:
    kps_pi_dt[pi] = {}
    for pid in kps_seq_pi_pid_np[pi]:
      dt = pi_pid_seq[pi][pid]['dt']
      kps_seq = kps_seq_pi_pid_np[pi][pid] # T x V x C

      for t in range(len(dt)):
        ts = dt[t]
        kps = kps_seq[t]

        if ts not in kps_pi_dt[pi]:
          kps_pi_dt[pi][ts] = {'pid': [], 'kps': []}
        kps_pi_dt[pi][ts]['pid'].append(pid)
        kps_pi_dt[pi][ts]['kps'].append(kps)
  # pprint (kps_pi_dt)
  # assert False

  degree_ep6_pi_pid = {}
  for pi in kps_pi_dt:
    degree_ep6_pi_pid[pi] = {}
    for ts in kps_pi_dt[pi]:
      year = ts.year
      month = ts.month
      day = ts.day
      hour = ts.hour
      minute = ts.minute
      second = ts.second

      dir_pi = _cfg.dir_posenet_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
      dir_vid = dir_pi + '/videos'
      file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'

      if os.path.exists(file_frame):
        kps_seq = np.array(kps_pi_dt[pi][ts]['kps'])
        min_x = np.amin(kps_seq[:,:,0], axis=1)
        min_y = np.amin(kps_seq[:,:,1], axis=1)
        max_x = np.amax(kps_seq[:,:,0], axis=1)
        max_y = np.amax(kps_seq[:,:,1], axis=1)

        w = max_x - min_x
        h = max_y - min_y

        bbox_seq = np.concatenate((min_y.reshape((-1,1)),
                                   min_x.reshape((-1,1)),
                                   h.reshape((-1,1)),
                                   w.reshape((-1,1))),
                                   axis=1)
        
        data_numpy = cv2.imread(file_frame, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        data_numpy = cv2.flip(data_numpy, 1)

        batch = []
        for i in range(bbox_seq.shape[0]):
          bbox = bbox_seq[i] # bbox: x, y, w, h

          c, s = dataloader._box2cs(bbox)

          trans = get_affine_transform(c, s, 0, dataloader.image_size)
          input = cv2.warpAffine(
              data_numpy,
              trans,
              (int(dataloader.image_size[0]), int(dataloader.image_size[1])),
              flags=cv2.INTER_LINEAR)
          if dataloader.transform:
              input = dataloader.transform(input)
          input = input.float()
          input = torch.unsqueeze(input, 0)
          batch.append(input)
        batch = torch.cat(batch, 0)

        _, hoe_output = model(batch)
        hoe_output= hoe_output.detach().cpu().numpy()
        index_degree = hoe_output.argmax(axis = 1)
        degree = -index_degree*5

        pids = kps_pi_dt[pi][ts]['pid']
        for i in range(len(pids)):
          pid = pids[i]
          degree_i = degree[i]

          if _cfg.viz_mebow_img:
            viz_mebow_img(batch[i], degree[i],
                          data_numpy, kps_seq[i], bbox_seq[i],
                          pi, pid, ts, _cfg)

          if pid not in degree_ep6_pi_pid[pi]:
            degree_ep6_pi_pid[pi][pid] = []
          degree_ep6_pi_pid[pi][pid].append(degree_i)
      else:
        pids = kps_pi_dt[pi][ts]['pid']
        for i in range(len(pids)):
          pid = pids[i]

          if pid not in degree_ep6_pi_pid[pi]:
            degree_ep6_pi_pid[pi][pid] = []
          degree_ep6_pi_pid[pi][pid].append(None)
    
  for pi in degree_ep6_pi_pid:
    for pid in degree_ep6_pi_pid[pi]:
      degree_seq = degree_ep6_pi_pid[pi][pid]
      # print (degree_seq)

      degree_seq = Series(degree_seq).interpolate()
      degree_seq = Series(degree_seq).fillna(method='ffill')
      degree_seq = Series(degree_seq).fillna(method='bfill')
      degree_seq = degree_seq.to_numpy()

      degree_ep6_pi_pid[pi][pid] = degree_seq
      # print (degree_seq)
    # assert False

  ori_ep6_pi_pid = {}
  for pi in degree_ep6_pi_pid:
    ori_ep6_pi_pid[pi] = {}
    for pid in degree_ep6_pi_pid[pi]:
      degree_seq = degree_ep6_pi_pid[pi][pid]

      ref_vec = np.array([0,1])
      radian = np.radians(degree_seq)
      cos, sin = np.cos(radian), np.sin(radian)
        
      T = degree_seq.shape[0]
      orientation = np.empty((T, 2))
      for t in range(T):
        c = cos[t]
        s = sin[t]
        R = np.array([[c, -s],
                      [s,  c]])
        orientation[t] = ref_vec @ R
      ori_ep6_pi_pid[pi][pid] = orientation
      
  return ori_ep6_pi_pid, degree_ep6_pi_pid