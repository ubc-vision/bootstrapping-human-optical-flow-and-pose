import sys, os
sys.path.append(os.path.abspath(os.path.join("..")))
from optimization_h36m import *
DEVICE = 'cuda'
device = torch.device('cuda')
#error lists
e1 = []
e2 = []

def load_image_metro(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    return img

def intermediates(p1, p2, nb_points=8):
    """"returns a list of nb_points equally spaced points
    between p1 and p2"""
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

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]


    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def get_flow(res_frame, pj2d_prev, pj2d_curr):
    """returns optical flow values at the DCPose joint locations"""
    flo = res_frame[9]
    flow_up_u = flo[:,:,0]
    flow_up_v = flo[:,:,1]
    u = torch.zeros((17,1),device = device)
    v = torch.zeros((17,1),device = device)
    row = flow_up_u.shape[0]
    col = flow_up_u.shape[1]
    for i in range(1):
        m = pj2d_prev
        for l in range(17):
            uy = m[l][1].item()
            ux = m[l][0].item()
            if ux<col and uy<row:
                
                if ux == 0.0 and uy == 0.0:
                    u[l] = pj2d_curr[l][0] - pj2d_prev[l][0]
                    v[l] = pj2d_curr[l][1] - pj2d_prev[l][1]
                else:
                    u[l] = bilinear_interpolate(flow_up_u, ux, uy)
                    v[l] = bilinear_interpolate(flow_up_v, ux, uy)
            else:
                u[l] = pj2d_curr[l][0] - pj2d_prev[l][0]
                v[l] = pj2d_curr[l][1] - pj2d_prev[l][1]
        uv = torch.cat((u, v), axis=1)
    return uv

def get_joints_dcpose_h36m(file_num, s):
        """returns 2d detections and confidence values from off the shelf 2D pose estimator DCPose, 
           rearrangement is done so that it matches METRO joint ordering"""
        from .flow_utils import read_json_from_file
        dcpose_json = read_json_from_file("dcpose_h36m/"+s+"/result_"+file_num+".json")
        x_det = np.zeros((len(dcpose_json['Info']),1, 14, 2))
        confidence = np.zeros((len(dcpose_json['Info']),1, 14, 1))
        for im in range(len(dcpose_json['Info'])):
            joint_mapper = [-100, -100, -100, -100, -100, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
            for keypoint in range(len(dcpose_json['Info'][im]['keypoints'])):
                if keypoint != 0 and keypoint != 1 and keypoint != 2 and keypoint != 3 and keypoint != 4 and keypoint != 11 and keypoint != 12  :
                    x_det[im][0][joint_mapper[keypoint]][0] = dcpose_json['Info'][im]['keypoints'][keypoint][0]
                    x_det[im][0][joint_mapper[keypoint]][1] = dcpose_json['Info'][im]['keypoints'][keypoint][1]
                    #if both keypoints x,y are zero, then set the confidence of the keypoints to zero
                    if dcpose_json['Info'][im]['keypoints'][keypoint][0] == 0 and dcpose_json['Info'][im]['keypoints'][keypoint][1] == 0: 
                        confidence[im][0][joint_mapper[keypoint]][0] = 0
                    else:
                        confidence[im][0][joint_mapper[keypoint]][0] = 1 
        return  x_det, confidence

def get_label(file_num, res, s):
        """returns the annotations/gts after aligning the joints with pelvis"""
        results = np.load('data/data_3d_h36m.npz',allow_pickle=True)['positions_3d'].item()
        x_label = np.zeros((len(res), 1, 54, 3))
        counter1 = 1
        for x in range(x_label.shape[0]):
            if counter1 >= (results[s][file_num].shape[0]):
                break
            for human in range(x_label.shape[1]):
                results[s][file_num][counter1] -= results[s][file_num][counter1][11]
                pelvis = (results[s][file_num][counter1][1] + results[s][file_num][counter1][6]) / 2.
                z = [55, 45, 5, 8, 55, 55, 46, 4, 7, 55, 55, 55, 55, 47, 55, 48, 55, 16, 18, 20, 55, 55, 55, 55, 55, 17, 19, 21, 55, 55, 55, 55]
                for l in range(32):
                    if l!=0 and l != 4 and l != 5 and l != 9 and l != 10 and l != 11 and l != 12 and l != 14 and l != 16 and l != 20 and l != 21 and l != 22 and l != 23 and l != 24 and l != 28 and l != 29 and l != 30 and l != 31:
                        x_label[x][human][z[l]][0] = results[s][file_num][counter1][l][0] - pelvis[0]
                        x_label[x][human][z[l]][1] = results[s][file_num][counter1][l][1] - pelvis[1]
                        x_label[x][human][z[l]][2] = results[s][file_num][counter1][l][2] - pelvis[2]            
            counter1 += 5 
        return x_label

def reorder_idx(results, x_label, file_num,s):
        """returns the reordered optimized and original estimates. The estimates are also aligned with the pelvis joint"""
        x_orig = np.zeros((x_label.shape))
        x_optim = np.zeros((x_label.shape))
        for frame in range(x_label.shape[0]):
            for human in range(x_label.shape[1]): 
                pelvis_optim = (results[(frame)][0]['pred_3d_joints_from_smpl_optim'][0][2] + results[(frame)][0]['pred_3d_joints_from_smpl_optim'][0][3]) / 2.
                pelvis_orig = (results[(frame)][0]['pred_3d_joints_from_smpl'][0][2] + results[(frame)][0]['pred_3d_joints_from_smpl'][0][3]) / 2.
                for l in range(14):
                            z = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48]
                            x_optim[frame][human][z[l]][0] = results[(frame)][0]['pred_3d_joints_from_smpl_optim'][0][l][0] - pelvis_optim[0]
                            x_optim[frame][human][z[l]][1] = results[(frame)][0]['pred_3d_joints_from_smpl_optim'][0][l][1] - pelvis_optim[1]
                            x_optim[frame][human][z[l]][2] = results[(frame)][0]['pred_3d_joints_from_smpl_optim'][0][l][2] - pelvis_optim[2]
                            x_orig[frame][human][z[l]][0] = results[(frame)][0]['pred_3d_joints_from_smpl'][0][l][0] - pelvis_orig[0]
                            x_orig[frame][human][z[l]][1] = results[(frame)][0]['pred_3d_joints_from_smpl'][0][l][1] - pelvis_orig[1]
                            x_orig[frame][human][z[l]][2] = results[(frame)][0]['pred_3d_joints_from_smpl'][0][l][2] - pelvis_orig[2]
        return x_orig, x_optim
    
def compute_error_helper(results, file_num, s):
        """returns the annotations, optimized estimates and original estimates after reordering and aligning the indices"""
        x_label = get_label(file_num, results, s)
        x_orig, x_optim = reorder_idx(results, x_label, file_num, s)
        return x_label, x_orig, x_optim

    
    
def compute_error(results, file_num, s):
        """computes the MPJPE"""
        x_label, x_orig, x_optim = compute_error_helper(results, file_num, s)
        for x in range(x_label.shape[0]): 
            optim_error = 0
            orig_error = 0
            joint_count = 0
            is_all_zero = np.all((x_optim[x] == 0))
            if is_all_zero == False : 
                for human in range(x_label.shape[1]):
                    for l in range(x_label.shape[2]):
                        if x_label[x][human][l][0] != 0 or x_label[x][human][l][1] != 0 or x_label[x][human][l][2] != 0:
                            optim_error += math.sqrt((x_optim[x][human][l][2] - x_label[x][human][l][2])**2 + (x_optim[x][human][l][0] - x_label[x][human][l][0])**2 + (x_optim[x][human][l][1] - x_label[x][human][l][1])**2)
                            orig_error += math.sqrt((x_orig[x][human][l][2] - x_label[x][human][l][2])**2 + (x_orig[x][human][l][0] - x_label[x][human][l][0])**2 + (x_orig[x][human][l][1] - x_label[x][human][l][1])**2)
                            joint_count += 1
                if joint_count != 0:
                    e1.append(optim_error*1000/joint_count)
                    e2.append(orig_error*1000/joint_count)        
        

def project(X_original, results, cam):
    """projects 3D joints onto the 2D image and returns the projected values"""
    kp_original_all = torch.zeros(X_original.shape[0], 1, 14, 2, device = device)
    
    for i in range(X_original.shape[0]):
        camera = cam[i].reshape(-1, 1, 3)
        X = X_original[i]
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        shape = X_trans.shape
        x = camera[:, :, 0] * X_trans.reshape(shape[0], -1)
        project_cropped = x.reshape(shape)
        project_cropped = ((project_cropped + 1) * 0.5) * 224
        #---------------------------------------------------
        joints = project_cropped[0]
        img_size = results[i][0]['proc_param']['img_size']
        undo_scale = 1. / np.array(results[i][0]['proc_param']['scale'])
        principal_pt = np.array([img_size, img_size]) / 2.
        start_pt = results[i][0]['proc_param']['start_pt'] - 0.5 * img_size
        final_principal_pt = (principal_pt + start_pt) * undo_scale
        margin = int(img_size / 2)
        kp_original = (joints + torch.tensor(results[i][0]['proc_param']['start_pt'], device=device) - margin) * undo_scale
        kp_original_all[i] = kp_original 
    return kp_original_all


def get_flow_metro(res_frame, pj2d_prev):
    """returns optical flow values at the 2D projections of 3D METRO joint locations"""
    flo = res_frame[0]['flow']
    flow_up_u = flo[:,:,0]
    flow_up_v = flo[:,:,1]
    u = np.zeros((14,1))
    v = np.zeros((14,1))
    row = flow_up_u.shape[0]
    col = flow_up_u.shape[1]
    for i in range(1):
        m = pj2d_prev
        for l in range(14):
            uy = m[l][1].item()
            ux = m[l][0].item()
            if ux<col and uy<row:
                if ux.is_integer() and uy.is_integer():
                    u[l] = flow_up_u[int(uy)][int(ux)]
                    v[l] = flow_up_v[int(uy)][int(ux)]
                else:
                    u[l] = bilinear_interpolate(flow_up_u, ux, uy)
                    v[l] = bilinear_interpolate(flow_up_v, ux, uy)
            else:
                u[l] = bilinear_interpolate(flow_up_u, ux, uy)
                v[l] = bilinear_interpolate(flow_up_v, ux, uy)
        uv = np.concatenate((u, v), axis=1)
    return uv
                
def compute_bone_length(X):
    """computes bone lengths of the bones in the skeleton for all the frames in the video"""
    pdist = torch.nn.PairwiseDistance(p = 2)
    skel = torch.zeros((X.shape[0],15))
    joint_pairs_indices = [[0,1], [1,2], [2,8],[2,3], [5,4], [4,3], [3,9], [12, 8], [12, 9], [9, 10], [10, 11], [8, 7], [7, 6], [12,13],[9,8]]
    count = 0
    for bone in joint_pairs_indices:
        skel[:,count] = pdist(X[:, 0, bone[0]], X[:, 0, bone[1]])
        count += 1
    return skel