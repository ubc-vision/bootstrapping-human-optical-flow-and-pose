# Bootstrapping human optical flow and pose
This repository contains the code for our paper:

[Bootstrapping human optical flow and pose](https://arxiv.org/abs/2210.15121)

BMVC 2022

**Requirements**
```
conda create --name metro_raft
conda activate metro_raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```
**Required data**

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
