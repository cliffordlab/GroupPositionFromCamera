import sys
sys.path.append('/home/chegde/projects/CEP/EP6_vision_testing/group_behavior_analysis/social_interaction/mebow')
import importlib
from graph.coco import Graph
import numpy as np
import torch

def ori_est_mebow(kps_seq_pi_pid_np, cfg):

  device = torch.device('cuda', cfg.cude_device_id)

  # load model
  _module = importlib.import_module(f'models.{cfg.mebow_model}')
  Model = _module.Model

  model = Model(graph=Graph(), 
                 classification=cfg.mebow_classification)

  file_mebow_model = cfg.file_mebow_model + f'/{cfg.mebow_model}'
  if cfg.mebow_classification:
    file_mebow_model += '_classification'
  file_mebow_model += '/best.pth'
  best_state = torch.load(file_mebow_model)
  model.load_state_dict(best_state['model_state_dict'])
  print ('load from ...', file_mebow_model)
                
  model.to(device)
  print (model)
  assert False

  model.eval()
  
  ori_ep6_pi_pid = {}
  degree_ep6_pi_pid = {}
  for pi in kps_seq_pi_pid_np:
    ori_ep6_pi_pid[pi] = {}
    degree_ep6_pi_pid[pi] = {}
    for pid in kps_seq_pi_pid_np[pi]:
      kps_seq = kps_seq_pi_pid_np[pi][pid] # T x V x C
      T, V, C = kps_seq.shape
      # print (kps_seq.shape)

      max_xy = np.amax(kps_seq, axis=1) # T x C
      # print (max_xy.shape)
      max_xy = max_xy.reshape((T, 1, C))
      # print (max_xy.shape)
      min_xy = np.amin(kps_seq, axis=1) # T x C
      # print (min_xy.shape)
      min_xy = min_xy.reshape((T, 1, C))
      # print (min_xy.shape)

      range_xy = max_xy - min_xy
      range_xy[range_xy==0] = 1.

      kps_seq = (kps_seq - min_xy)/range_xy
      # print (np.amin(kps_seq), np.amax(kps_seq))
      kps_seq = kps_seq.reshape((T, V, 1, C)) # T x V x 1 x C
      # print (kps_seq.shape)
      kps_seq = np.transpose(kps_seq, axes=[0,3,2,1]) # T x C x 1 X V
      # print (kps_seq.shape)

      kps_seq = torch.from_numpy(kps_seq).to(device, dtype=torch.float)
      pred = model(kps_seq)
      _, pred = torch.max(pred, 1)
      pred = pred.detach().cpu().numpy()
      # print (pred.shape)
      # assert False

      # degree = -pred * 5
      degree = pred * 5 # just trying ...
      degree_ep6_pi_pid[pi][pid] = degree

      ref_vec = np.array([0,1])
      radian = np.radians(degree)
      cos, sin = np.cos(radian), np.sin(radian)
      # print (degree.shape)
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

      ori_ep6_pi_pid[pi][pid] = orientation
      # assert False
  
  return ori_ep6_pi_pid, degree_ep6_pi_pid




