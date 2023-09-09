#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:33:41 2023

@author: chegde
"""

import sys
from argparse import Namespace
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
from mebow.dataset import COCO_HOE_Dataset_Img
import importlib
from graph.coco import Graph

def load_mebow_model(_cfg):
    
    if _cfg.mebow_image:
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
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
        dataloader = COCO_HOE_Dataset_Img(cfg,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize,]),
                                    flip_hori=True)
    else:
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
        
        dataloader = 'NA'
    
    return model, dataloader