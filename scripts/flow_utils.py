import sys, os
sys.path.append(os.path.abspath(os.path.join("..")))
from optimization_h36m import *
DEVICE = 'cuda'
device = torch.device('cuda')
err1 = []
err2 = []
err3 = []
total_frame = []


def load_model(args, file_num):
    """loads the RAFT model. Returns the model."""
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    return model


def load_flo(path):
    """loads the GT optical flow and returns it."""
    flo = glob.glob(os.path.join(path, '*.flo'))
    flo = sorted(flo)
    flo_gt = {}
    for flow_num in range(len(flo)):
        path = flo[flow_num]
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h, w, 2))
        flo_gt[flow_num] = data2D 
    return flo_gt

def read_json_from_file(input_path):
    """reads json file and returns it."""
    import json
    with open(input_path, "r") as read_file:
        python_data = json.load(read_file)
    return python_data

def load_image(imfile):
    """loads image and returns it."""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points between p1 and p2."""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)
    points = np.zeros((nb_points+2,2))
    points[0][0] = p1[0]
    points[0][1] = p1[1]
    points[points.shape[0]-1][0] = p2[0]
    points[points.shape[0]-1][1] = p2[1]
    for i in range(1, nb_points+1):
        points[i][0] = p1[0] + i * x_spacing
        points[i][1] = p1[1] +  i * y_spacing
    return points

def get_new_flowmap(results, skel, confidence, s=""):
            """Creates a flowmap for the skeleton pixel locations and overlays on top of the optical flow network's flow estimates. Returns the target flowmaps"""
            #if DCPose
            if s != "S9" and s != "S11":
                new_results = np.zeros((len(results), results[0][9].shape[0], results[0][9].shape[1], results[0][9].shape[2]), dtype = float)
                for frame in range(len(results)-1):
                    new_results[frame] = results[frame][9].copy()
                
                for frame in range(1, skel.shape[0]):
                    is_all_zero = np.all((confidence[frame] == 0))
                    is_all_zero_prev = np.all((confidence[frame-1] == 0))    
                    if True:
                        for l in range(skel.shape[2]):
                                if round(skel[frame-1][0][l][1]) < results[0][9].shape[0] and round(skel[frame-1][0][l][0])<results[0][9].shape[1] and is_all_zero == False and is_all_zero_prev == False and skel[frame][0][l][0]!=0 and skel[frame-1][0][l][0]!=0 and skel[frame][0][l][1]!=0 and skel[frame-1][0][l][1]!=0:
                                    new_results[frame-1][round(skel[frame-1][0][l][1])-neighborhood : round(skel[frame-1][0][l][1])+neighborhood, round(skel[frame-1][0][l][0])-neighborhood : round(skel[frame-1][0][l][0])+neighborhood, 0] = skel[frame][0][l][0] - skel[frame-1][0][l][0]
                                    new_results[frame-1][round(skel[frame-1][0][l][1])-neighborhood : round(skel[frame-1][0][l][1])+neighborhood, round(skel[frame-1][0][l][0])-neighborhood : round(skel[frame-1][0][l][0])+neighborhood, 1] = skel[frame][0][l][1] - skel[frame-1][0][l][1] 
            #if 2D projections of METRO
            else:
                new_results = np.zeros((len(results), results[0][0]['flow'].shape[0], results[0][0]['flow'].shape[1], results[0][0]['flow'].shape[2]), dtype = float)
                for frame in range(len(results)-1):
                    new_results[frame] = results[frame][0]['flow']   
                for frame in range(1, skel.shape[0]):
                    for l in range(skel.shape[2]):
                            if skel[frame][0][l][0]!=0 and skel[frame-1][0][l][0]!=0 and skel[frame][0][l][1]!=0 and skel[frame-1][0][l][1]!=0:
                                new_results[frame-1][round(skel[frame-1][0][l][1])-neighborhood : round(skel[frame-1][0][l][1])+neighborhood, round(skel[frame-1][0][l][0])-neighborhood : round(skel[frame-1][0][l][0])+neighborhood, 0] = skel[frame][0][l][0] - skel[frame-1][0][l][0]
                                new_results[frame-1][round(skel[frame-1][0][l][1])-neighborhood : round(skel[frame-1][0][l][1])+neighborhood, round(skel[frame-1][0][l][0])-neighborhood : round(skel[frame-1][0][l][0])+neighborhood, 1] = skel[frame][0][l][1] - skel[frame-1][0][l][1]
            new_results = torch.tensor(new_results)
            return new_results  

def generate_flowmap(file_num, x_det, confidence, results, loop_count, s = ""):
      """Generates target flow map from 2D joint estimates. Returns target flow map after generating the skeleton."""
      #if DCPose
      if s != "S9" and s != "S11":
        #if first optimization cycle, load original estimates of DCPose. 
        #if second optimization cycle, we already have the optimized DCPose estimates from the pose optimization cycle.
        if loop_count == 0:
            dcpose_json = read_json_from_file("dcpose_sintel/result_"+file_num+".json")
            x_det = np.zeros((len(dcpose_json['Info']),1, 17, 2))
            confidence = np.zeros((len(dcpose_json['Info']),1, 17, 1))
            for im in range(len(dcpose_json['Info'])):
                for keypoint in range(len(dcpose_json['Info'][im]['keypoints'])):
                    if keypoint != 3 and keypoint != 4:
                        x_det[im][0][keypoint][0] = dcpose_json['Info'][im]['keypoints'][keypoint][0]
                        x_det[im][0][keypoint][1] = dcpose_json['Info'][im]['keypoints'][keypoint][1]
                        #if both keypoints x,y are zero, then set the confidence of the keypoints to zero
                        if dcpose_json['Info'][im]['keypoints'][keypoint][0] == 0 and dcpose_json['Info'][im]['keypoints'][keypoint][1] == 0: 
                            confidence[im][0][keypoint][0] = 0
                        else:
                            confidence[im][0][keypoint][0] = dcpose_json['Info'][im]['keypoints'][keypoint][2]
        #create skeleton from the 2D estimates stored in x_det.
        #joint combination for the bones.
        joint_pairs_indices = [[2,0],[0,1], [1,5], [5,7], [7,9], [1,6], [6,8], [8,10], [5,11], [11,13], [13,15], [6,12], [12,14], [14,16]]
        skel = np.zeros((x_det.shape[0],1, 14, nb_points+2, 2))
        for im in range(skel.shape[0]):
            for keypoint in range(skel.shape[2]):
                j_0 = joint_pairs_indices[keypoint][0]
                j_1 = joint_pairs_indices[keypoint][1]
                if confidence[im][0][j_0][0] > 0.6 and confidence[im][0][j_1][0] > 0.6: 
                    skel[im][0][keypoint] = intermediates([x_det[im][0][j_0][0], x_det[im][0][j_0][1]], [x_det[im][0][j_1][0], x_det[im][0][j_1][1]], nb_points)
        skel = skel.reshape(skel.shape[0], 1, skel.shape[2]*skel.shape[3],-1)
        #generate the target flowmap from 2D skeleton
        new_results  = get_new_flowmap(results, skel, confidence,s)
        target = new_results.float()
        return target, x_det, confidence
      # if METRO
      else:
        #create skeleton from the 2D projections of METRO.
        #joint combination for the bones.
        skel = np.zeros((len(results), 1, 13, nb_points+2, 2))
        joint_pairs_indices = [[0,1], [1,2], [2,8], [5,4], [4,3], [3,9], [12, 8], [12, 9], [9, 10], [10, 11], [8, 7], [7, 6], [12,13]]
        for im in range(len(results)):
            for keypoint in range(skel.shape[2]):
                j_0 = joint_pairs_indices[keypoint][0]
                j_1 = joint_pairs_indices[keypoint][1]
                skel[im][0][keypoint] = intermediates([results[im][0]['joints_optim'][0][j_0][0], results[im][0]['joints_optim'][0][j_0][1]], [results[im][0]['joints_optim'][0][j_1][0], results[im][0]['joints_optim'][0][j_1][1]], nb_points)
        skel = skel.reshape(skel.shape[0], 1, skel.shape[2]*skel.shape[3],-1)
        new_results = get_new_flowmap(results, skel, None, s)
        return new_results, None, None

def compute_error_flow(results, file_num, cycle_num, s=""):
    """computes optical flow error, EPE"""
    if s != "S9" and s != "S11": 
        path = 'data/sintel/training/flow/' + file_num
        flo_gt = load_flo(path)
        flo_pred = np.zeros((len(results), results[0][9].shape[0], results[0][9].shape[1], 2))
        for frame in range(len(results)-1):
            flo_pred[frame] = results[frame][9] 
    else:
        flo_gt, flo_pred = gt_flow(results, file_num, s)
    if cycle_num == 1:
        if s != "S9" and s != "S11":
            for i in range(len(flo_gt)):
                    error1 = torch.sum((torch.tensor(flo_pred[i]).cuda() - torch.tensor(flo_gt[i]).cuda())**2, dim=2).sqrt()
                    err1.append(error1.view(-1).cpu().numpy()) 
        else:
            f1 = []
            for i in range(len(flo_gt)-1):
                    error1 = torch.sum((torch.tensor(flo_pred[i]).cuda() - torch.tensor(flo_gt[i]).cuda())**2, dim=2).sqrt()
                    f1.append(error1.view(-1).cpu().numpy())   
            err1.append(np.sum(np.concatenate(f1)))                    
                
    elif cycle_num == 0:
        if s != "S9" and s != "S11":
            for i in range(len(flo_gt)):
                    error2 = torch.sum((torch.tensor(flo_pred[i]).cuda() - torch.tensor(flo_gt[i]).cuda())**2, dim=2).sqrt()
                    err2.append(error2.view(-1).cpu().numpy())
        else:    
            f1 = []
            for i in range(len(flo_gt)-1):
                    error1 = torch.sum((torch.tensor(flo_pred[i]).cuda() - torch.tensor(flo_gt[i]).cuda())**2, dim=2).sqrt()
                    f1.append(error1.view(-1).cpu().numpy())   
            total_frame.append(len(f1)) 
            err2.append(np.sum(np.concatenate(f1)))                    
                              
    else:
        if s != "S9" and s != "S11":
            for i in range(len(flo_gt)):
                error1 = torch.sum((torch.tensor(flo_pred[i]).cuda() - torch.tensor(flo_gt[i]).cuda())**2, dim=2).sqrt()
                err3.append(error1.view(-1).cpu().numpy())
        else:    
            f1 = []
            for i in range(len(flo_gt)-1):
                    error1 = torch.sum((torch.tensor(flo_pred[i]).cuda() - torch.tensor(flo_gt[i]).cuda())**2, dim=2).sqrt()
                    f1.append(error1.view(-1).cpu().numpy())   
            err3.append(np.sum(np.concatenate(f1)))                    
                 

def infer(model, results, args, mode_check, s=""):
    """runs inference on the data"""
    sys.path.append('RAFT/core')
    from raft import RAFT
    from utils import flow_viz
    from utils.utils import InputPadder
    if mode_check == 1:
        model = model.train()
    else:
        model = model.eval()
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                        glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)
        count = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow_up = padder.unpad(flow_up)
                if s!="S9" and s!="S11":
                    results[count][9] = flow_up[0].permute(1,2,0).detach().cpu().numpy()
                else:
                    results[count][0]['flow'] = flow_up[0].permute(1,2,0).detach().cpu().numpy()
                count+=1         
    return results



def get_label_2d(file_num, s):
        file_id = int(file_num.split(".")[1])
        file_name = file_num.split(".")[0]
        if file_id == 54138969:
            file_id = 0
        if file_id == 55011271:
            file_id = 1
        if file_id == 58860488:
            file_id = 2
        if file_id == 60457274:
            file_id = 3    
        results = np.load('data/h36m/data_2d_h36m_gt.npz',allow_pickle=True)['positions_2d'].item()
        x_label_2d = results[s][file_name][file_id]
        x_label_2d_new = np.zeros((x_label_2d.shape))
        frame_id = 0
        for orig_frame_id in range(x_label_2d.shape[0]):
            if (orig_frame_id-1) % 5 == 0:
                x_label_2d_new [frame_id] =  x_label_2d [orig_frame_id]
                frame_id += 1
        return x_label_2d_new

def gt_flow(results, path, s):
    x_label_2d = get_label_2d(path, s)
    flo_gt = np.zeros((len(results), results[0][0]['flow'].shape[0], results[0][0]['flow'].shape[1], 2))
    flo_pred = np.zeros((len(results), results[0][0]['flow'].shape[0], results[0][0]['flow'].shape[1], 2))
    for frame in range(1, len(results)):
            for l in range(x_label_2d.shape[1]):
                            flo_gt[frame-1][round(x_label_2d[frame-1][l][1])][round(x_label_2d[frame-1][l][0])][0] = x_label_2d[frame][l][0] - x_label_2d[frame-1][l][0]
                            flo_gt[frame-1][round(x_label_2d[frame-1][l][1])][round(x_label_2d[frame-1][l][0])][1] = x_label_2d[frame][l][1] - x_label_2d[frame-1][l][1]
                            flo_pred[frame-1][round(x_label_2d[frame-1][l][1])][round(x_label_2d[frame-1][l][0])][0] = results[frame-1][0]['flow'][round(x_label_2d[frame-1][l][1])][round(x_label_2d[frame-1][l][0])][0]
                            flo_pred[frame-1][round(x_label_2d[frame-1][l][1])][round(x_label_2d[frame-1][l][0])][1] = results[frame-1][0]['flow'][round(x_label_2d[frame-1][l][1])][round(x_label_2d[frame-1][l][0])][1]
    return flo_gt, flo_pred


