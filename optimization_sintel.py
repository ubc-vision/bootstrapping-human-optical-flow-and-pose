from numpy import dot
from numpy.linalg import norm
from matplotlib import image 
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
from statistics import mean 
from argparse import Namespace
import cv2
import os
import glob
import torch
import random
import numpy as np
import  pickle
import math
import argparse
import sys
from scripts.flow_utils import *
from scripts.pose_utils import *

sys.path.append('RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
DEVICE = 'cuda'
device = torch.device('cuda')

#hyperparameters
pose_steps = 1500
flow_steps = 50
nb_points = 20
lambda_flow = 300
lambda_2D = 000
lambda_init = 400
neighborhood = 3      

def flow_optimization(results, x_det, confidence, file_num, model, loop_count):
      """Optical flow optimization using pose"""
      
      args = Namespace(alternate_corr = False, corr_levels = 4, corr_radius = 4, dropout = 0, mixed_precision = True, model = 'RAFT/models/raft-things.pth', path = 'data/sintel/'+file_num, small=False)        
      if loop_count == 0:
        #load the raft model and evaluate its performance in terms of End Point Error (EPE)
        model = load_model(args, file_num)
        results = infer(model, results, args, 0)
        compute_error_flow(results, file_num, 0)

      #--------------------------------------------------
      #START OF FLOW OPTIMIZATION
      
      #get the initial estimates
      results = infer(model, results, args, 1)
      model = model.train()
      #generate target flow map
      target, x_det, confidence = generate_flowmap(file_num, x_det, confidence, results, loop_count)
      #Huber loss and Adam optimizer
      criterion = torch.nn.SmoothL1Loss()
      optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
      #finetuning loop starts
      for step in range(flow_steps):
            if True:
                images = glob.glob(os.path.join(args.path, '*.png')) + \
                        glob.glob(os.path.join(args.path, '*.jpg'))
                images = sorted(images)
                count = 0
                loss_list = []            
                for imfile1, imfile2 in zip(images[:-1], images[1:]):
                      optimizer.zero_grad() 
                      image1 = load_image(imfile1)
                      image2 = load_image(imfile2)
                      padder = InputPadder(image1.shape)
                      image1, image2 = padder.pad(image1, image2)
                      flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                      flow_up = padder.unpad(flow_up)   
                      l = criterion(flow_up[0].permute(1,2,0).cuda(), (target[count]).cuda()).sum()
                      loss_list.append(l.float())
                      l.backward()
                      optimizer.step()
                      count += 1
            
            print("Step- ", step)
            print('Loss', sum(loss_list)/len(loss_list))
      #finetuning loop ends
      #evaluate the results of the fine tuned model
      results = infer(model, results, args, 0)
      if loop_count == 0:
        compute_error_flow(results, file_num, 1)
        print("Current averages (first optim, orig) : ", np.mean(np.concatenate(err1)), np.mean(np.concatenate(err2)))        
      else:
        compute_error_flow(results, file_num, 2)
        print("Current averages (first optim, second optim, orig) : ", np.mean(np.concatenate(err1)), np.mean(np.concatenate(err3)), np.mean(np.concatenate(err2)))
      return results, x_det, confidence, model


def pose_optimization(results, file_num, file_path, x_det, confidence):
      #optimizes the 2D pose based on temporal information and optical flow
      #------------------------------------------------------------------------------
      #creating tensors for training/optimizing
      X_tensor = torch.tensor(x_det, device = device)
      X_init = torch.tensor(x_det, device = device)
      
      #Huber loss and Adam optimizer
      criterion = torch.nn.SmoothL1Loss(reduction='none')
      optimizer = torch.optim.Adam([X_tensor], 0.1)
      X_tensor.requires_grad = True
      stop_criterion = 0
      prev_loss = -1
      #finetuning loop starts
      for step in range(pose_steps):
            #initializing the loss with zero
            loss = torch.zeros(1, requires_grad=True)
            loss_list = []
            optimizer.zero_grad()
            count = 0
            for k in range(len(results)):
                for human in range(1):
                        is_all_zero = np.all((confidence[count] == 0))
                        is_all_zero_prev = np.all((confidence[count-1] == 0))
                        if count != 0 and is_all_zero == False and is_all_zero_prev == False:
                                #temporal loss
                                temp_l =  lambda_2D * criterion(X_tensor[(count)][human].float(), X_tensor[(count-1)][human].float()).sum()
                                #flow loss
                                flow_l = lambda_flow * criterion((get_flow(results[(count-1)], X_tensor[(count-1)][human], X_tensor[(count)][human])).float(),(X_tensor[(count)][human] - X_tensor[(count-1)][human]).float()).sum()                             
                        else:
                            #temporal loss
                            temp_l = 0
                            #flow loss
                            flow_l = 0
                        #prior term
                        prior_l = lambda_init * criterion(X_tensor[(count)][human], X_init[(count)][human]).float().sum()
                        l  = temp_l + prior_l + flow_l
                        loss_list.append(l.float())                                    
                count += 1
            loss = sum(loss_list)/(len(loss_list))
            loss.backward()
            optimizer.step()
            if format(loss.item(), ".4f") == format(prev_loss, ".4f"):
                stop_criterion += 1
            else:
                stop_criterion = 0
            if stop_criterion == 10:
                print(step)
                break
            prev_loss = loss.item()
      #finetuning loop ends
      #optimized 2D estimates stored in x_det_opt
      x_det_opt = X_tensor.detach().cpu().numpy().astype('float32')
      return x_det_opt, confidence


def data_paths_full():
      import os
      import os.path as path
      rootdir = 'data/sintel/training/final'
      new_files = []
      for subdir, dirs, files in os.walk(rootdir):
        if subdir == rootdir:
            continue            
        new_files.append(subdir)
      return new_files  

def main():
        file_path = data_paths_full()
        for i in range(len(file_path)):
            file_num = file_path[(i)].split("/")[-1]
            if file_num == 'alley_2' or file_num == 'bamboo_2' or file_num == 'cave_2' or file_num == 'cave_4':
                
                print("Optimizing - ",file_num)
                print("-------------")
                results = np.load('metro_estimates_sintel/' + file_num +'_results.npz',allow_pickle=True)['results'][()]
                results, x_det, confidence, model = flow_optimization(results, None, None, file_num, None, 0)
                x_det, confidence = pose_optimization(results, file_num, file_path[i], x_det, confidence)
                results, x_det, confidence, model = flow_optimization(results, x_det, confidence, file_num, model, 1)
                    
if __name__ == '__main__':
    main()

