# Bootstrapping human optical flow and pose
This repository contains the code for our paper:

[Bootstrapping human optical flow and pose](https://arxiv.org/abs/2210.15121)

BMVC 2022

## Requirements
```
conda create --name metro_raft
conda activate metro_raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```
## Required data

To evaluate/run our optimization pipeline, you will need to download the required dataset.

* [Human3.6M](http://vision.imar.ro/human3.6m/description.php)

The directory should look like this - 
```bash
├── data
    ├── h36m
        ├── S9
            ├── Videos
        ├── S11
            ├── Videos
```    

## Preprocess Human3.6M dataset and get the 2D and 3D annotation files

The preprocessed 2D and 3D annotation files data_2d_h36m_gt.npz and data_3d_h36m.npz are generated by using the data preparation code in [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://github.com/facebookresearch/VideoPose3D). The generated files should be copied to "data/h36m/". It can be done manually or you can simply run:

```
cp /path/to/file/data_2d_h36m_gt.npz data/h36m/
cp /path/to/file/data_3d_h36m.npz data/h36m/
```

## Pre-computed Mesh Transformer (METRO) and Deep Dual Consecutive Network for Human Pose Estimation (DCPose) estimates

Our pre-computed METRO and DCPose estimates must be downloaded in order to run the framework. This can be done by simply downloading the folders [metro_estimates_h36m](https://drive.google.com/drive/folders/1w4lOmWpRwNDm88B__YB3_w2pcatcFwu0?usp=share_link) and 
[dcpose_estimates_h36m](https://drive.google.com/drive/folders/13Js77b5LjDC1YzEJh268jfZjGWeZkpM3?usp=share_link).

## Finetuning human optical flow and pose using our framework

The code for finetuning human optical flow and pose on the Human3.6M can be found in optimization_h36m.py file. We show optimization on top of [Mesh Transformer (METRO)](https://github.com/microsoft/MeshTransformer) and [Recurrent All Pairs Field Transforms for Optical Flow (RAFT)](https://github.com/princeton-vl/RAFT).
 
 
