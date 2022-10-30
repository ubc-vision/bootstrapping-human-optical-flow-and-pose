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
lambda_flow = 0.01
lambda_3D = 300
lambda_det = 0.01
lambda_2D = 0.001
lambda_cam = 0.1
lambda_bone = 10000
lambda_init = 400
pose_steps = 1500
flow_steps = 8
neighborhood = 15
nb_points = 20

def pose_optimization(results, file_num, s, loop_counter, base_input_image_path_raft, input_video_file_path):
      """optimizes the 3D pose based on temporal information, bone length information and optical flow"""
      X = np.zeros((len(results),1, 14, 3), dtype = 'float32')
      X_init_numpy = np.zeros((len(results),1, 14, 3), dtype = 'float32')
      cam = np.zeros((len(results),1, 1, 3), dtype = 'float32')
      #filling the initialized arrays with METRO-RAFT estimates
      if loop_counter == 0:
            steps = 0   
            for frame in range(len(results)):
                    X[frame] = results[frame][0]['pred_3d_joints_from_smpl'].cpu().numpy()
                    cam[frame][0][0] = results[frame][0]['pred_camera'].cpu().numpy()
      else:
            steps = pose_steps 
            for frame in range(len(results)):
                X[frame] = results[frame][0]['pred_3d_joints_from_smpl_optim']
                cam[frame][0][0] = results[frame][0]['camera_optim'].cpu().numpy()
      
      X_init_numpy = X
      #get 2D keypoints and confidence values from DCPose
      x_det, confidence = get_joints_dcpose_h36m(file_num, s)          
      #creating tensors for training/optimizing
      X_tensor = torch.tensor(X, device = device)
      X_init = torch.tensor(X_init_numpy, device = device)
      cam_tensor = torch.tensor(cam, device = device)      
    
      #Huber loss and Adam optimizer
      criterion = torch.nn.SmoothL1Loss(reduction='none')
      optimizer = torch.optim.Adam([X_tensor, cam_tensor], 0.001)
      X_tensor.requires_grad = True
      cam_tensor.requires_grad = True
      stop_criterion = 0
      prev_loss = -1
      #finetuning loop starts
      for step in range(steps):
            #project 3D joints on to 2D image for all the frames in the video
            pj2d = project(X_tensor, results, cam_tensor)
            #compute bone length of skeletons in all the frames
            skel = compute_bone_length(X_tensor)
            #initialize loss
            loss = torch.zeros(1, requires_grad=True)
            loss_list = []
            optimizer.zero_grad()
            count = 0
            for k in range(len(results)):
                for human in range(1):
                    pj2d_curr = pj2d[count]
                    #if not first frame
                    if count != 0:
                        pj2d_prev = pj2d[count-1]
                        #temporal loss
                        temp_l = lambda_cam * criterion(cam_tensor[count], cam_tensor[count-1]).sum().float() + lambda_2D * criterion(pj2d_curr, pj2d_prev).sum().float()  + lambda_3D * criterion(X_tensor[count][human], X_tensor[(count-1)][human]).sum().float() + lambda_bone * criterion(skel[count], skel[count-1]).sum().float()
                        #flow loss
                        flow_l = lambda_flow * criterion(torch.tensor(get_flow_metro(results[(count-1)], pj2d_prev[human]), device = device).float(),(pj2d_curr[human] - pj2d_prev[human]).float()).sum().float()
                    #if first frame    
                    else:
                        #temporal loss
                        temp_l = 0
                        #flow loss
                        flow_l = 0
                    #reprojection loss
                    det_l = lambda_det * criterion(torch.tensor(confidence[count][human], device = device) * torch.tensor(x_det[count][human], device = device), torch.tensor(confidence[count][human], device = device) * pj2d_curr[human]).sum().float()
                    #prior term
                    prior_l = lambda_init * criterion(X_tensor[count][human], X_init[count][human]).sum().float()
                    l  = temp_l +  det_l + prior_l + flow_l
                    loss_list.append(l.float())                                    
                count += 1
            loss = sum(loss_list)/(len(loss_list))
            if step % 100 == 0:
                pass
                # print("Step- ", step)
                # print('Loss', loss)
            loss.backward()
            optimizer.step()
            
            if format(loss.item(), ".4f") == format(prev_loss, ".4f"):
                stop_criterion += 1
            else:
                stop_criterion = 0
            if stop_criterion == 10:
                break
            prev_loss = loss.item()
      
      #store optimized values in the "results" dictionary
      X_tensor_copy = X_tensor
      projection = project((X_tensor_copy), results, cam_tensor).detach().cpu().numpy().astype('float32')
      for frame in range(len(results)):
            if loop_counter == 0:
                results[frame][0]['pred_3d_joints_from_smpl'] = X_tensor[frame].detach()
                results[frame][0]['joints_orig'] = projection[frame]
            results[frame][0]['pred_3d_joints_from_smpl_optim'] = X_tensor[frame].detach().cpu().numpy().astype('float32')
            results[frame][0]['joints_optim'] = projection[frame]
            results[frame][0]['camera_optim'] = (cam_tensor[frame].detach())
      
      if loop_counter == 0:
        pass
        get_video_frames("data/h36m/" + s + input_video_file_path + file_num + ".mp4", results, base_input_image_path_raft)
      return results


def flow_optimization(results, x_det, confidence, file_num, model, loop_count, s=""):
      """Optical flow optimization using pose"""
      
      args = Namespace(alternate_corr = False, corr_levels = 4, corr_radius = 4, dropout = 0, mixed_precision = True, model = 'RAFT/models/raft-things.pth', path = 'raft_inputs/', small=False)        
      if loop_count == 0:
        #load the raft model and evaluate its performance in terms of End Point Error (EPE)
        model = load_model(args, file_num)
        results = infer(model, results, args, 0, s)
        compute_error_flow(results, file_num, 0, s)
        
      #--------------------------------------------------
      #START OF FLOW OPTIMIZATION
      
      #get the initial estimates
      results = infer(model, results, args, 1, s)
      model = model.train()
      #generate target flow map
      target, _, _ = generate_flowmap(file_num, x_det, confidence, results, loop_count, s)
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
            if (step+1)%4==0:
                pass
                # print("Step- ", step)
                # print('Loss', sum(loss_list)/len(loss_list))
      #finetuning loop ends
      #evaluate the results of the fine tuned model
      results = infer(model, results, args, 0, s)
      if loop_count == 0:
        compute_error_flow(results, file_num, 1, s)  
      else:
        compute_error_flow(results, file_num, 2, s)
        print("Current EPE (first optimization, second optimization, original) : ", sum((err1))/(32*sum(total_frame)), sum((err3))/(32*sum(total_frame)), sum((err2))/(32*sum(total_frame)))
      return results, model, x_det, confidence



def get_video_frames(video_file_path, results, base_input_image_path_raft):
    from scripts.utility import OpenCVCapture
    capture = OpenCVCapture(video_file_path)
    video_length = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #remove all the images in the input folder of RAFT
    r_files = glob.glob(base_input_image_path_raft+"*")
    for r_f in r_files:
        os.remove(r_f)
    
    frame_id = 0
    for orig_frame_id in range(video_length):
        if frame_id >= len(results):
            break
        frame = capture.read()
        if (orig_frame_id-1) % 5 == 0:
            plt.imsave(base_input_image_path_raft+str(frame_id).zfill(4)+'.jpg', frame)
            frame_id += 1

def data_paths_full(s, input_video_file_path):
      import os
      import os.path as path
      file_path, dirs, files = next(os.walk("data/h36m/" + s + input_video_file_path))
      return files  


def main():
    subjects = ["S9", "S11"]
    base_input_image_path_raft = 'raft_inputs/'
    input_video_file_path = '/Videos/'
    for s in subjects:
        file_path = data_paths_full(s, input_video_file_path)
        print("Total number of videos, ", len(file_path))
        for i in range(len(file_path)):
            file_num = file_path[(i)].split(".mp4")[0]
            if  file_num.split(".")[-1] == "60457274" and file_num != '_ALL.60457274': 
                print("Optimizing - ",file_num)
                results = np.load('metro_estimates_h36m/'+s+'/'+ file_num +'_results.npz',allow_pickle=True)['results'][()]
                results = pose_optimization(results, file_num, s, 0, base_input_image_path_raft, input_video_file_path)
                print("First cycle of optical flow optimization begins")
                results, model, _, _ = flow_optimization(results, None, None, file_num, None, 0, s)
                print("First cycle of pose optimization begins")
                results = pose_optimization(results, file_num, s, 1, base_input_image_path_raft, input_video_file_path)
                print("Second cycle of optical flow optimization begins")
                results, model, _, _ = flow_optimization(results, None, None, file_num, model, 1, s)
                compute_error(results, file_path[(i)].split(".mp4")[0], s)            
                print("Current MPJPE (first optimization, original) : ", sum(e1)/len(e1), sum(e2)/len(e2))
                print("Done optimizing ",file_num)
                print("----------------------------------")
                
if __name__ == '__main__':
    main()

