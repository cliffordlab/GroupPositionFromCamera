from argparse import Namespace
import sys
sys.path.append('/home/hhyeokk/Research/EP6/code/EP6_vision_testing/social_interaction')
from pandas import Series
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import os
from mebow.dataset import COCO_HOE_Dataset_Img
from debug import viz_mebow_img

#def ori_est_mebow_img(kps_seq_pi_pid_np, pi_pid_seq, _cfg):
def ori_est_mebow_img(kps_seq_pi_pid_np, pi_pid_seq, _cfg):

  sys.path.append(_cfg.path_mebow_repo)
  #-------------------
  from lib import models
  from lib.config import cfg, update_config
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

  ori_ep6_pi_pid = {}
  degree_ep6_pi_pid = {}
  for pi in kps_seq_pi_pid_np:
    ori_ep6_pi_pid[pi] = {}
    degree_ep6_pi_pid[pi] = {}
    for pid in kps_seq_pi_pid_np[pi]:
      
      dt = pi_pid_seq[pi][pid]['dt']
      kps_seq = kps_seq_pi_pid_np[pi][pid] # T x V x C
      # print (pi, pid, len(dt), kps_seq)
      # assert False

      if 0:
        # looks like x and y axis is flipped?
        min_xy = np.amin(kps_seq, axis=1)
        max_xy = np.amax(kps_seq, axis=1)

        xy_seq = min_xy
        wh_seq = max_xy - min_xy
        bbox_seq = np.concatenate((xy_seq, wh_seq), axis=1)
      else:
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
      # print (bbox_seq)
      # assert False

      T = len(dt)

      degree_seq = []
      for i in range(bbox_seq.shape[0]):
        bbox = bbox_seq[i]

        ts = dt[i]
        year = ts.year
        month = ts.month
        day = ts.day
        hour = ts.hour
        minute = ts.minute
        second = ts.second

        dir_pi = _cfg.dir_data +  f'/{year}/{month:02}/{day:02d}/hour_{hour:02d}' + f'/{pi}'
        dir_vid = dir_pi + '/videos'
        file_frame = dir_vid + f'/{pi}{year}{month:02}{day:02}_{hour:02}{minute:02}{second:02d}.jpg'
        if os.path.exists(file_frame):
          input = dataloader(file_frame, bbox)
          input = torch.unsqueeze(input, 0)
          _, hoe_output = model(input)
          hoe_output= hoe_output.detach().cpu().numpy()
          index_degree = hoe_output.argmax(axis = 1)
          degree = -index_degree[0] * 5
          degree_seq.append(degree)

          # TODO: viz
          if _cfg.viz_mebow_img:
            viz_mebow_img(input, degree,
                          file_frame, kps_seq[i], bbox,
                          pi, pid, ts, _cfg)


        else:
          degree_seq.append(None)
              
      # print (degree_seq)
      degree_seq = Series(degree_seq).interpolate()
      degree_seq = Series(degree_seq).fillna(method='ffill')
      degree_seq = Series(degree_seq).fillna(method='bfill')
      degree_seq = degree_seq.to_numpy()
      # print (degree_seq)
      # assert False

      degree_ep6_pi_pid[pi][pid] = degree_seq

      ref_vec = np.array([0,1])
      radian = np.radians(degree_seq)
      cos, sin = np.cos(radian), np.sin(radian)
      # print (degree_seq.shape)
      # print (cos.shape)
      # print (sin.shape)
      
      orientation = np.empty((T, 2))
      for t in range(T):
        c = cos[t]
        s = sin[t]
        R = np.array([[c, -s],
                      [s,  c]])
        orientation[t] = ref_vec @ R
      # print (orientation.shape)
      # print (orientation.shape)
      # print (orientation)
      # assert False

      ori_ep6_pi_pid[pi][pid] = orientation

  return ori_ep6_pi_pid, degree_ep6_pi_pid