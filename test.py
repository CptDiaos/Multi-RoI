"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm
from collections import OrderedDict


import config
import constants
from models import hmr, SMPL, SMPL_agora
from datasets import BaseDataset, CustomDataset
from utils.geometry import cam_crop2full
from utils.imutils import uncrop, flip_img
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
from utils.renderer import Renderer
from utils.renderer_pytorch3d import Renderer_Pytorch3d as R3d

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='hmr', help='name of exp model')
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', 'lsp-orig', '3dpw', 'mpi-inf-3dhp','coco', 'agora', 'custom'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=8, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npy file')
parser.add_argument('--viz', default=False, action='store_true', help='Visualize the mesh result')
parser.add_argument('--use_latent', default=False, action='store_true', help='Use latent encoder')
parser.add_argument('--use_pseudo', default=False, action='store_true', help='Use pseudo labels')
parser.add_argument('--out_num', type=int, default=3, help='Number of ouptput value') 
parser.add_argument('--use_extraviews', default=False, action='store_true', help='Use latent encoder')
parser.add_argument('--use_fuse', default=False, action='store_true', help='Use pseudo labels')
parser.add_argument('--bbox_type', default='square', help='Use square bounding boxes in 224x224 or rectangle ones in 256x192')
parser.add_argument('--rescale_bbx', default=False, action='store_true', help='Use rescaled bbox for consistency data aug and loss')
parser.add_argument('--shift_center', default=False, action='store_true', help='Use shifted center for consistency data aug and loss')
parser.add_argument('--n_views', type=int, default=5, help='Views to use') 
parser.add_argument('--rand', default=False, action='store_true', help='Use random augs')
parser.add_argument('--large_bbx', default=False, action='store_true', help='Use large bounding box')
parser.add_argument('--encoder', default='hr32', help='The type of backbone encoder')
parser.add_argument('--folder', default='ours-hr48', help='which method')
parser.add_argument('--interval', type=int, default=5, help='interval for viz')
parser.add_argument('--mesh_npy', default=None, help='directory for meshes')
parser.add_argument('--use_r3d', default=False, action='store_true', help='if there exists available prediction results')
parser.add_argument('--val', default=False, action='store_true', help='if there exists available prediction results')

global_args = parser.parse_args()

def run_test(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=16, shuffle=False, log_freq=50, 
                   with_train=False, eval_epoch=None, summary_writer=None, out_num=3, viz=False, model_aux=None, use_extra=False, use_fuse=False, n_views=5, bbox_type='square', folder='ours-hr48', interval=5, use_r3d=False, signated=False, val=False):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    aroundy = cv2.Rodrigues(np.array([0, np.radians(-90.), 0]))[0]

    # Transfer model to the GPU

    if model_aux is not None:
        model_aux.to(device)
    else:
        model.to(device)
    print(bbox_type)
    if bbox_type=='square':
        crop_w = 224.
        crop_h = 224.
        bbox_size = 224.
    elif bbox_type=='rect':
        crop_w = 192.
        crop_h = 256.
        bbox_size = 256.
    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    val_suffix = "_val" if val else ''
    
    # renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    
    if use_r3d:
        faces_tensor = torch.from_numpy(smpl_neutral.faces.astype(np.int32)).to(device)
    if folder=='fastmetro':
        res_dict = np.load('/media/lab345/Elements SE/lcz/data/fastmetro/fastmetro_result_{}.npy'.format(dataset_name), allow_pickle=True).item()
        imgnames_ = res_dict['imgnames']
        ori_idx = np.array([int(x.split('/')[-1].split('_')[0]) for x in imgnames_])
        sorted_idx = ori_idx.argsort()
        use_res=True
        pred_vertices_ = np.array(res_dict['pred_vertices'])[sorted_idx]
        pred_joints_ = np.array(res_dict['pred_joints'])[sorted_idx]
        pred_cams_ = np.array(res_dict['pred_cams'])[sorted_idx]
        print(pred_vertices_.shape)
    # imgnames_ = res_dict['imgnames'][sorted_idx]

    elif folder=='refit':
        dict_path = '/media/lab345/Elements SE/lcz/data/refit/refit_result_{}{}'.format(dataset_name, val_suffix)
        if 'h36m' not in dataset_name:
            if dataset_name == '3dpw':
                dict_path += '_aligned'
            # res_dict = np.load('../ReFit-main/refit_result_{}{}_aligned.npz'.format(dataset_name, val), allow_pickle=True).item()
            res_dict = np.load('{}.npz'.format(dict_path), allow_pickle=True)
            use_res=True
            pred_rotmats_ = torch.from_numpy(np.array(res_dict['pose'])).to(device).to(torch.float32)
            pred_cams_ = torch.from_numpy(np.array(res_dict['camera'])).to(device).to(torch.float32)
            pred_trans_ = torch.from_numpy(np.array(res_dict['trans'])).to(device).to(torch.float32)
        else:
            pred_cams_ = torch.from_numpy(np.array(res_dict['camera'])).to(device).to(torch.float32)
            pred_vertices_ = res_dict['pred_vertices']
    else:
        use_res = False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == 'coco' or dataset_name == 'agora':
        eval_pose = True

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    all_idx = 0
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        bbox_info = batch['bbox_info'].to(device)
        curr_batch_size = images.shape[0]
        img_h = batch['img_h'].to(device)
        img_w = batch['img_w'].to(device)
        focal_length = batch['focal_length'].to(device)
        img_names = batch['imgname']
        center = batch['center'].to(device)
        scale = batch['scale'].to(device)
        sc = batch['scale_factor'].to(device)
        rot = batch['rot_angle'].to(device)
        gt_keypoints_2d = batch['keypoints'].to(device)
        gt_keypoints_2d_full = batch['keypoints_full'].to(device)
        is_flipped = batch['is_flipped'].to(device)
        has_smpl = batch['has_smpl'].byte().to(device)  # flag that indicates whether SMPL parameters are valid
        # pseudo_imgs = batch['pseudo_img'].to(device)
        pseudo_imgs = images.clone()
        rot_rad = rot / 180. * torch.pi
        if use_extra:
            bboxes_info_extra = batch['bboxes_info_extra'].to(device)
            imgs_extra = batch['imgs_extra'].to(device)
            images = torch.cat([images, imgs_extra], 1)
            bbox_info = torch.cat([bbox_info, bboxes_info_extra], 1)
        
        idx=0
        if not use_res:
            with torch.no_grad():
                print('The number of outputs: {}'.format(out_num))
                if int(out_num) == 3:
                    pred_rotmat, pred_betas, pred_camera = model(images, bbox_info)

                    if use_fuse:
                        pred_rotmat = pred_rotmat.view(curr_batch_size, n_views, 24, 3, 3)[:, 0]
                        pred_betas = pred_betas.view(curr_batch_size, n_views, -1)[:, 0]
                        pred_camera = pred_camera.view(curr_batch_size, n_views, -1)[:, 0]
                elif int(out_num) == 4:
                    pred_rotmat, pred_betas, pred_camera, _1 = model(images, bbox_info)
                    if use_fuse:
                        print("Using {}th view to calculate".format(idx))
                        pred_rotmat = pred_rotmat.view(curr_batch_size, n_views, 24, 3, 3)[:, idx]
                        pred_betas = pred_betas.view(curr_batch_size, n_views, -1)[:, idx]
                        pred_camera = pred_camera.view(curr_batch_size, n_views, -1)[:, idx]
                elif int(out_num) == 5:
                    if model_aux is not None:
                        pred_rotmat, pred_betas, pred_camera, _1, _2 = model_aux(pseudo_imgs, bbox_info)
                    else:
                        pred_rotmat, pred_betas, pred_camera, _1, _2 = model(images, bbox_info)
                elif int(out_num) == 6:
                    pred_rotmat, pred_betas, pred_camera, _1, _2, _3 = model(images, bbox_info)
                elif int(out_num) == -1:
                    pred_rotmat_list, pred_betas_list, pred_camera_list, _1, _2 = model(images, bbox_info)
                    pred_rotmat = pred_rotmat_list[-1]
                    pred_betas = pred_betas_list[-1]
                    pred_camera = pred_camera_list[-1]

                pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices
                # pred_vertices_neutral = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
                # pred_vertices = smpl_male(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
                # pred_vertices_female = smpl_female(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
                # pred_vertices[gender==1, :, :] = pred_vertices_female[gender==1, :, :]

        else:
            if signated:
                imgnames = imgnames_[step * batch_size:step * batch_size + curr_batch_size]
                if '3760_outdoors_fencing_01-image_00098.jpg.jpg' not in imgnames and '17490_S11_Discussion_2.60457274_001891.jpg' not in imgnames:
                    # print(imgnames[0])
                    continue
                else:
                    sign_idx = imgnames.index('3760_outdoors_fencing_01-image_00098.jpg.jpg') if dataset_name =='3dpw' else imgnames.index('17490_S11_Discussion_2.60457274_001891.jpg')
                    print(sign_idx)
            if folder =='fastmetro':
                pred_vertices = torch.from_numpy(pred_vertices_[step * batch_size:step * batch_size + curr_batch_size]).to(device)
                pred_camera = torch.from_numpy(pred_cams_[step * batch_size:step * batch_size + curr_batch_size]).to(device)
            elif folder=='refit':
                if 'h36m' not in dataset_name:
                    pred_rotmats = pred_rotmats_[step * batch_size:step * batch_size + curr_batch_size]
                    # print(pred_rotmats.shape)
                    rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
                    rotmat = torch.cat((pred_rotmats.reshape(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
                    pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
                    pred_output = smpl_female(betas=gt_betas, body_pose=pred_pose[:, 3:], global_orient=pred_pose[:, :3])
                    pred_vertices = pred_output.vertices + pred_trans_[step * batch_size:step * batch_size + curr_batch_size]
                else:
                    pred_vertices = torch.from_numpy(np.array(pred_vertices_[step * batch_size:step * batch_size + curr_batch_size])).to(device).to(torch.float32)
                pred_camera = pred_cams_[step * batch_size:step * batch_size + curr_batch_size]
                J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
                pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
                pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
                # pred_vertices = pred_vertices - pred_pelvis
        
        if viz:
            multi_viz = False
            color_dict = {
                # 'ours-hr48' : (1.0, 0.5, 0.),
                'refit': (0.8,  0.60,  0.9, 1.0),
                'fastmetro' : (0.35, 0.60, 0.92, 1.0),
                'cliff-hr48' : (1., 0.2, 0.5, 1.0),
                'ours-hr48' : (0.99, 0.9, 0.4, 1.0),}
            img_dir = 'test_result/{}/{}/'.format(folder, dataset_name)
            print(img_dir)
            if not os.path.isdir(img_dir):
                os.makedirs(img_dir)
            
            pred_cam_crop = torch.stack([pred_camera[:, 1],
                                     pred_camera[:, 2],
                                     2 * 5000 / (bbox_size * pred_camera[:, 0] + 1e-9)],
                                    dim=-1)
            full_img_shape = torch.stack((img_h, img_w), -1)
            pred_cam_full = cam_crop2full(pred_camera, center, scale, full_img_shape,
                                    focal_length).to(torch.float32)
            if folder!='refit':
                    pred_cam_full = cam_crop2full(pred_camera, center, scale, full_img_shape,
                                        focal_length).to(torch.float32)
            else:
                pred_cam_full = torch.zeros_like(pred_cam_crop, device=device)
            
            if not use_extra:
                n_views = 1
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1).repeat(1, n_views, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1).repeat(1, n_views, 1, 1)
            # images = images[:, :3]
            # print(images.shape)
            # for idx in tqdm(np.random.choice(len(img_names), 4, replace=False)):
            if not use_r3d:
                for idx in tqdm(range(len(img_names))[::interval]):
                    if folder=='ours':
                        n_views_viz = n_views
                    else:
                        n_views_viz = 1
                    for view in range(n_views_viz):
                        images_v = images[:, 3*view:3*view+3]
                        if dataset_name=='3dpw':
                            name = '-'.join(img_names[idx].split('/')[-2:])
                        else:
                            name = '.'.join(img_names[idx].split('/')[-1].split('.')[:-1])
                        print(name)
                        renderer = Renderer(focal_length=focal_length[idx],
                                        img_res=[img_w[idx], img_h[idx]], faces=smpl_neutral.faces)
                        # print(crop_w, crop_h)
                        crop_renderer = Renderer(focal_length=5000,
                                            img_res=[crop_w, crop_h], faces=smpl_neutral.faces)
                        image_mat = np.ascontiguousarray((images_v[idx]*255.0).permute(1, 2, 0).cpu().detach().numpy())
                        # for kp in gt_keypoints_2d[idx, 25:]:
                        #     cv2.circle(image_mat, (int(kp[0]),int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                        
                        rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
                        # if dataset_name == '3dpw':
                        #     rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
                        # for kp in gt_keypoints_2d_full[idx, 25:]:
                        #     cv2.circle(rgb_img, (int(kp[0]),int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                        # rgb_img = np.full((int(img_h[idx]), int(img_w[idx]), 3), fill_value=255., dtype=np.float32)
                        if is_flipped[idx]:
                            rgb_img = flip_img(rgb_img)
                        if dataset_name == '3dpw':
                            rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
                        MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
                        rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))), (2, 0, 1)) / 255.0
                        image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
                        images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, base_color=color_dict[folder], grid=False, blank_bg=False)
                        # images_side = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, base_color=color_dict[folder], grid=False, blank_bg=True, side=aroundy)
                        
                        # images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images_v[idx:idx+1], grid=False, amb_color=(1, 1, 1), blank_bg=True)
                        cv2.imwrite('test_result/{}/{}{}/{}_{}_visualized.jpg'.format(folder, dataset_name, val_suffix, all_idx, name), np.ascontiguousarray(images_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                        # cv2.imwrite('test_result/{}/{}/{}_{}_side_visualized.jpg'.format(folder, dataset_name, all_idx, name), np.ascontiguousarray(images_side * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                        
                        # cv2.imwrite('test_result/{}/{}/{}_crop_visualized_{}.jpg'.format(folder, dataset_name, all_idx, name), np.ascontiguousarray(images_crop_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                        # cv2.imwrite('test_result/{}/{}/{}_full_origin.jpg'.format(folder, dataset_name, name), rgb_img[:, :, ::-1])
                        # cv2.imwrite('test_result/{}/{}/{}_crop_origin_{}.jpg'.format(folder, dataset_name, name, view), image_mat[:, :, ::-1])
                    all_idx += interval

            else:
                for idx in tqdm(range(len(img_names))[::interval]):
                    if signated and idx!=sign_idx:
                        continue
                    r3d = R3d(focal_length=focal_length[idx],
                                        img_res=[int(img_w[idx]), int(img_h[idx])], faces=faces_tensor)
                    crop_r3d = R3d(focal_length=5000,
                                            img_res=[int(crop_w), int(crop_h)], faces=faces_tensor)
                    if folder=='ours-hr48':
                        n_views_viz = 1
                    else:
                        n_views_viz = 1
                    for view in range(n_views_viz):
                        images_v = images[:, 3*view:3*view+3].permute(0, 2, 3, 1)*255.
                        if dataset_name=='3dpw':
                            name = '-'.join(img_names[idx].split('/')[-2:])
                        else:
                            name = '.'.join(img_names[idx].split('/')[-1].split('.')[:-1])
                        print("Visualizing....", name, view)
                        rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
                        if dataset_name == '3dpw':
                            rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
                        image_mat = np.ascontiguousarray((images_v[idx]).cpu().detach().numpy())
                        image_full = torch.from_numpy(rgb_img).to(images.device).unsqueeze(0)
                        images_pred = r3d.render_mesh(pred_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, base_color=color_dict[folder], grid=False, blank_bg=False)
                        # images_side = r3d.render_mesh(pred_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, base_color=color_dict[folder], grid=False, blank_bg=True, side=aroundy)
                        # images_crop_pred = crop_r3d.render_mesh(pred_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images_v[idx:idx+1], base_color=color_dict[folder], grid=False, blank_bg=False)
                        
                        # print(images_pred.shape)

                        
                        cv2.imwrite('test_result/{}/{}{}/{}_{}_visualized.jpg'.format(folder, dataset_name, val_suffix, all_idx, name), np.ascontiguousarray(images_pred)[:, :, ::-1])
                        # cv2.imwrite('test_result/{}/{}/{}_{}_side_visualized.jpg'.format(folder, dataset_name, all_idx, name), np.ascontiguousarray(images_side)[:, :, ::-1])
                        # cv2.imwrite('test_result/{}/{}{}/view_{}_{}_{}_crop_visualized.jpg'.format(folder, dataset_name, val_suffix, view, all_idx, name), np.ascontiguousarray(images_crop_pred)[:, :, ::-1])
                        # cv2.imwrite('test_result/{}/{}{}/view_{}_{}_{}_crop_origin.jpg'.format(folder, dataset_name, val_suffix, view, all_idx, name), image_mat[:, :, ::-1])
                    all_idx += interval
                

        if save_results:
            # rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            # rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            # pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            # smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            # smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            # if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
            if 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                print("Using smpl joints as gt 3d joints!", has_smpl[0])
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
                torch.set_printoptions(threshold=10000)
                # print(gender)
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                # print(gt_keypoints_3d.shape)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            # torch.set_printoptions(threshold=10000)
            # print(pred_keypoints_3d, gt_keypoints_3d)

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error (PA-MPJPE)
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Vertices to vertices (PVE)
            if '3dpw' in dataset_name:
                v_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                pve[step * batch_size:step * batch_size + curr_batch_size] = v_error


        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, pred_camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        # np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        np.save(result_file, res_dict)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

    if with_train:
        summary_writer.add_scalar('MPJPE', 1000 * mpjpe.mean(), eval_epoch)
        summary_writer.add_scalar('PA-MPJPE', 1000 * recon_err.mean(), eval_epoch)
        summary_writer.add_scalar('PVE', 1000 * pve.mean(), eval_epoch)
        return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()
    


def run_test_agora(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=16, shuffle=False, log_freq=50, 
                   with_train=False, eval_epoch=None, summary_writer=None, out_num=3, viz=False, model_aux=None, use_extra=False, use_fuse=False, n_views=5, bbox_type='square'):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU

    if model_aux is not None:
        model_aux.to(device)
    else:
        model.to(device)
    print(bbox_type)
    if bbox_type=='square':
        crop_w = 224.
        crop_h = 224.
        bbox_size = 224.
    elif bbox_type=='rect':
        crop_w = 192.
        crop_h = 256.
        bbox_size = 256.
    # Load SMPL model
    smpl_neutral = SMPL_agora(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL_agora(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL_agora(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == 'coco' or dataset_name == 'agora':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        bbox_info = batch['bbox_info'].to(device)
        curr_batch_size = images.shape[0]
        img_h = batch['img_h'].to(device)
        img_w = batch['img_w'].to(device)
        focal_length = batch['focal_length'].to(device)
        img_names = batch['imgname']
        center = batch['center'].to(device)
        scale = batch['scale'].to(device)
        sc = batch['scale_factor'].to(device)
        rot = batch['rot_angle'].to(device)
        gt_keypoints_2d = batch['keypoints'].to(device)
        gt_keypoints_2d_full = batch['keypoints_full'].to(device)
        is_flipped = batch['is_flipped'].to(device)
        has_smpl = batch['has_smpl'].byte().to(device)  # flag that indicates whether SMPL parameters are valid
        # pseudo_imgs = batch['pseudo_img'].to(device)
        pseudo_imgs = images.clone()
        rot_rad = rot / 180. * torch.pi
        if use_extra:
            bboxes_info_extra = batch['bboxes_info_extra'].to(device)
            imgs_extra = batch['imgs_extra'].to(device)
            images = torch.cat([images, imgs_extra], 1)
            bbox_info = torch.cat([bbox_info, bboxes_info_extra], 1)
        
        idx=0
        with torch.no_grad():
            print('The number of outputs: {}'.format(out_num))
            if int(out_num) == 3:
                pred_rotmat, pred_betas, pred_camera = model(images, bbox_info)

                if use_fuse:
                    pred_rotmat = pred_rotmat.view(curr_batch_size, n_views, 24, 3, 3)[:, 0]
                    pred_betas = pred_betas.view(curr_batch_size, n_views, -1)[:, 0]
                    pred_camera = pred_camera.view(curr_batch_size, n_views, -1)[:, 0]
            elif int(out_num) == 4:
                pred_rotmat, pred_betas, pred_camera, _1 = model(images, bbox_info)
                if use_fuse:
                    print("Using {}th view to calculate".format(idx))
                    pred_rotmat = pred_rotmat.view(curr_batch_size, n_views, 24, 3, 3)[:, idx]
                    pred_betas = pred_betas.view(curr_batch_size, n_views, -1)[:, idx]
                    pred_camera = pred_camera.view(curr_batch_size, n_views, -1)[:, idx]
            elif int(out_num) == 5:
                if model_aux is not None:
                    pred_rotmat, pred_betas, pred_camera, _1, _2 = model_aux(pseudo_imgs, bbox_info)
                else:
                    pred_rotmat, pred_betas, pred_camera, _1, _2 = model(images, bbox_info)
            elif int(out_num) == 6:
                pred_rotmat, pred_betas, pred_camera, _1, _2, _3 = model(images, bbox_info)
            elif int(out_num) == -1:
                pred_rotmat_list, pred_betas_list, pred_camera_list, _1, _2 = model(images, bbox_info)
                pred_rotmat = pred_rotmat_list[-1]
                pred_betas = pred_betas_list[-1]
                pred_camera = pred_camera_list[-1]

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            # pred_vertices_neutral = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
            # pred_vertices = smpl_male(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
            # pred_vertices_female = smpl_female(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
            # pred_vertices[gender==1, :, :] = pred_vertices_female[gender==1, :, :]
            
        
        if viz:
            pred_cam_crop = torch.stack([pred_camera[:, 1],
                                     pred_camera[:, 2],
                                     2 * 5000 / (bbox_size * pred_camera[:, 0] + 1e-9)],
                                    dim=-1)
            full_img_shape = torch.stack((img_h, img_w), -1)
            pred_cam_full = cam_crop2full(pred_camera, center, scale, full_img_shape,
                                    focal_length).to(torch.float32)
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            # print(images.shape)
            num = 0
            # for idx in tqdm(np.random.choice(len(img_names), 4, replace=False)):
            for idx in tqdm(range(len(img_names))):
                name = img_names[idx].split('/')[-1].split('.')[0]
                print(name)
                renderer = Renderer(focal_length=focal_length[idx],
                                img_res=[img_w[idx], img_h[idx]], faces=smpl_neutral.faces)
                # print(crop_w, crop_h)
                crop_renderer = Renderer(focal_length=5000,
                                    img_res=[crop_w, crop_h], faces=smpl_neutral.faces)
                image_mat = np.ascontiguousarray((images[idx]*255.0).permute(1, 2, 0).cpu().detach().numpy())
                color_list = np.array([(0.8,  0.60,  0.9),(1., 0.2, 0.5),(0.99, 0.9, 0.4,),(0.35, 0.60, 0.92),(0.1, 0.9, 0.1)])
                for kp_i, kp in enumerate(gt_keypoints_2d[idx, :]):
                    cv2.circle(image_mat, (int(kp[0]),int(kp[1])), radius=3, color=255*color_list[kp_i%len(color_list)], thickness=-1)
                
                rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
                # if dataset_name == '3dpw':
                #     rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
                # for kp in gt_keypoints_2d_full[idx, 25:]:
                #     cv2.circle(rgb_img, (int(kp[0]),int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                # rgb_img = np.full((int(img_h[idx]), int(img_w[idx]), 3), fill_value=255., dtype=np.float32)
                if is_flipped[idx]:
                    rgb_img = flip_img(rgb_img)
                if dataset_name == '3dpw':
                    rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
                MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
                rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))), (2, 0, 1)) / 255.0
                image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
                images_pred = renderer.visualize_tb(gt_vertices[idx].unsqueeze(0), pred_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, grid=False)
                images_crop_pred = crop_renderer.visualize_tb(gt_vertices[idx].unsqueeze(0), pred_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images[idx:idx+1])
                cv2.imwrite('eval_result/img_viz/{}_{}_{}_img_visualized.jpg'.format(num,dataset_name, name), np.ascontiguousarray(images_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                cv2.imwrite('eval_result/img_viz/{}_{}_{}_img_crop_visualized.jpg'.format(num,dataset_name, name), np.ascontiguousarray(images_crop_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                cv2.imwrite('eval_result/img_viz/{}_{}_{}_img_full_origin.jpg'.format(num,dataset_name, name), rgb_img[:, :, ::-1])
                cv2.imwrite('eval_result/img_viz/{}_{}_{}_img_crop_origin.jpg'.format(num,dataset_name, name), image_mat[:, :, ::-1])
                num = num+1
        
        break
                

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            # if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
            if 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                print("Using smpl joints as gt 3d joints!", has_smpl[0])
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
                torch.set_printoptions(threshold=10000)
                # print(gender)
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                # print(gt_keypoints_3d.shape)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            # torch.set_printoptions(threshold=10000)
            # print(pred_keypoints_3d, gt_keypoints_3d)

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error (PA-MPJPE)
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Vertices to vertices (PVE)
            if '3dpw' in dataset_name:
                v_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                pve[step * batch_size:step * batch_size + curr_batch_size] = v_error


        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, pred_camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

    if with_train:
        summary_writer.add_scalar('MPJPE', 1000 * mpjpe.mean(), eval_epoch)
        summary_writer.add_scalar('PA-MPJPE', 1000 * recon_err.mean(), eval_epoch)
        summary_writer.add_scalar('PVE', 1000 * pve.mean(), eval_epoch)
        return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()



if __name__ == '__main__':
    args = global_args
    # if not args.use_latent:
    #     # model = hmr(config.SMPL_MEAN_PARAMS, name='hmr_sim', bbox_type=args.bbox_type)
    #     model = hmr(config.SMPL_MEAN_PARAMS, name='hmr_fuseviews')
    #     # out_num = 3
    #     out_num = 3
    # else:
    #     model = hmr(config.SMPL_MEAN_PARAMS, name='with_raw', att_num=args.att_num)
    #     out_num = 5
    result_file = os.path.join('logs', 'params.txt')
    logger = open(os.path.join(result_file), mode='w+', encoding='utf-8')
    model = hmr(config.SMPL_MEAN_PARAMS, name=args.model_name, bbox_type=args.bbox_type, encoder=args.encoder)
    # model = hmr(config.SMPL_MEAN_PARAMS, name=args.model_name)
    checkpoint = torch.load(args.checkpoint)
    ## Edit keys in original weights file from CLIFF
    # print(checkpoint.keys())
    # old_dict = checkpoint['state_dict']
    # old_dict = checkpoint['model']
    # new_dict = OrderedDict([(k.replace('module.', ''), v) for k,v in old_dict.items()])
    # weights = []
    # for k, v in old_dict.items():
    #     if k not in ["module.init_pose", "module.init_shape", "module.init_cam", "module.fc1.weight", "module.fc1.bias", "module.fc2.weight", "module.fc2.bias", "module.decpose.weight", \
    #                  "module.decpose.bias", "module.decshape.weight", "module.decshape.bias", "module.deccam.weight", "module.deccam.bias"]:
    #         weights.append((k.replace('module.encoder', 'backbone'), v))
    #     else:
    #         weights.append((k.replace('module.', ''), v))
    # new_dict = OrderedDict(weights)
    # new_dict = OrderedDict([(k.replace('model.', ''), v) for k,v in old_dict.items()])
    # new_dict = OrderedDict([(k.replace('head.', ''), v) for k,v in new_dict.items()])
    # new_dict = OrderedDict([(k.replace('encoder.', ''), v) for k,v in new_dict.items()])
    # for k,v in new_dict.items():
    #     print(k, v, file=logger)
    # model.load_state_dict(new_dict, strict=True)
    # model_dict = {}
    # model_dict['model'] = new_dict
    # torch.save(model_dict, args.checkpoint[:-3]+'_fitted.pt')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    # Setup evaluation dataset
    if args.dataset == 'custom':
        dataset = CustomDataset(args, args.dataset, is_train=False, bbox_type=args.bbox_type)
    else:
        dataset = BaseDataset(args, args.dataset, is_train=False, bbox_type=args.bbox_type, val=args.val)
    # dataset = BaseDataset(None, '3dpw', is_train=False, use_extraviews=True)
    # Run evaluation

    if args.dataset == 'agora':
        run_test_agora(model, args.dataset, dataset, args.result_file,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    log_freq=args.log_freq,
                    out_num=args.out_num,
                    viz=args.viz,
                    model_aux=None,
                    bbox_type=args.bbox_type,
                    use_extra=args.use_extraviews,
                    use_fuse=args.use_fuse,)
    else:
        run_test(model, args.dataset, dataset, args.result_file,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    log_freq=args.log_freq,
                    out_num=args.out_num,
                    viz=args.viz,
                    model_aux=None,
                    bbox_type=args.bbox_type,
                    use_extra=args.use_extraviews,
                    use_fuse=args.use_fuse,
                    folder=args.folder,
                    use_r3d=args.use_r3d,
                    interval=args.interval,
                    val=args.val
                    )
