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
import matplotlib.pyplot as plt
from collections import OrderedDict


import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.geometry import cam_crop2full
from utils.imutils import uncrop, flip_img
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
from utils.renderer import Renderer


# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='hmr', help='name of exp model')
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp','coco', 'agora'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=8, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
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
parser.add_argument('--by_category', default=False, action='store_true', help='calculate metrics by category')
parser.add_argument('--by_keypoint', default=False, action='store_true', help='calculate metrics by keypoint')
parser.add_argument('--use_res', default=False, action='store_true', help='if there exists available prediction results')
parser.add_argument('--val', default=False, action='store_true', help='if there exists available prediction results')


def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=16, shuffle=False, log_freq=50, 
                   with_train=False, eval_epoch=None, summary_writer=None, out_num=3, viz=False, model_aux=None, use_extra=False, use_fuse=False, n_views=5, bbox_type='square', category=None, keypoint=None, use_res=False):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    if model is not None:
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
    
    renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset


    if use_res:
        res_dict = np.load('../FastMETRO-main/fastmetro_result_h36m-p2.npy', allow_pickle=True).item()
        imgnames_ = res_dict['imgnames']
        ori_idx = np.array([int(x.split('/')[-1].split('_')[0]) for x in imgnames_])
        sorted_idx = ori_idx.argsort()
        # imgnames_ = res_dict['imgnames'][sorted_idx]
        pred_vertices_ = np.array(res_dict['pred_vertices'])[sorted_idx]
        pred_joints_ = np.array(res_dict['pred_joints'])[sorted_idx]
        print(pred_vertices_.shape)
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
    res_dict = {}
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))
    imgnames_ = []
    pred_vertices_ = np.zeros((len(dataset), 6890, 3))

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
                    pred_rotmat, pred_betas, pred_camera, _1 = model(images, bbox_info, n_extraviews=n_views-1)
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
            imgnames = imgnames_[step * batch_size:step * batch_size + curr_batch_size]
            pred_vertices = torch.from_numpy(pred_vertices_[step * batch_size:step * batch_size + curr_batch_size]).to(device)
            # print(len(img_names), img_names)
            # print(len(imgnames), imgnames)
            pass
        
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
            for idx in tqdm(np.random.choice(len(img_names), 4, replace=False)):
                name = img_names[idx].split('/')[-1].split('.')[0]
                print(name)
                renderer = Renderer(focal_length=focal_length[idx],
                                img_res=[img_w[idx], img_h[idx]], faces=smpl_neutral.faces)
                # print(crop_w, crop_h)
                crop_renderer = Renderer(focal_length=5000,
                                    img_res=[crop_w, crop_h], faces=smpl_neutral.faces)
                image_mat = np.ascontiguousarray((images[idx]*255.0).permute(1, 2, 0).cpu().detach().numpy())
                # for kp in gt_keypoints_2d[idx, 25:]:
                #     cv2.circle(image_mat, (int(kp[0]),int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                
                rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
                if dataset_name == '3dpw':
                    rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
                # for kp in gt_keypoints_2d_full[idx, 25:]:
                #     cv2.circle(rgb_img, (int(kp[0]),int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                # rgb_img = np.full((int(img_h[idx]), int(img_w[idx]), 3), fill_value=255., dtype=np.float32)
                if is_flipped[idx]:
                    rgb_img = flip_img(rgb_img)
                MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
                rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))), (2, 0, 1)) / 255.0
                image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
                # images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, grid=False)
                # images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images[idx:idx+1])
                images_pred = renderer.visualize_tb(gt_vertices[idx].unsqueeze(0), pred_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, grid=False, save_path='eval_result/img_viz/{}_{}_img_gt.obj'.format(dataset_name, num))
                images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images[idx:idx+1], grid=False, save_path='eval_result/img_viz/{}_{}_img_pred.obj'.format(dataset_name, num))
                cv2.imwrite('eval_result/img_viz/{}_img_visualized.jpg'.format(num), np.ascontiguousarray(images_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                cv2.imwrite('eval_result/img_viz/{}_img_crop_visualized.jpg'.format(num), np.ascontiguousarray(images_crop_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                # cv2.imwrite('eval_result/img_viz/{}_{}_img_full_origin.jpg'.format(dataset_name, num), rgb_img[:, :, ::-1])
                # cv2.imwrite('eval_result/img_viz/{}_{}_img_crop_origin.jpg'.format(dataset_name, num), image_mat[:, :, ::-1])
                num = num+1
                

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.reshape(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            # smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            pred_vertices_[step * batch_size:step * batch_size + curr_batch_size, :] = pred_vertices.cpu().numpy()
            imgnames_ += img_names
            # print(imgnames_, len(imgnames_))
            
        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            # if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
            if 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_pelvis = gt_keypoints_3d[:, [14],:-1].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 
                gt_vertices_aligned = gt_vertices - gt_pelvis
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
                gt_vertices_aligned = gt_vertices - gt_pelvis
                
            if viz:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                p_points = pred_vertices[0, :, :3].cpu().detach().numpy()
                g_points = gt_vertices[0, :, :3].cpu().detach().numpy()
                # for joint_idx in range(pred_keypoints_3d.shape[1]):
                ax.scatter3D(p_points[:, 0], p_points[:, 1], p_points[:, 2], c='green')
                ax.scatter3D(g_points[:, 0], g_points[:, 1], g_points[:, 2], c='red')
                    # ax.scatter3D(x=g_points[joint_idx, 0], y=g_points[joint_idx, 1], z=g_points[joint_idx, 2], c='red')
                plt.savefig('eval_result/img_viz/3D_vertices.jpg')
                plt.close()
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            # print(pred_keypoints_3d.shape)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_vertices_aligned = pred_vertices - pred_pelvis
            # if use_res:
            #     pred_keypoints_3d = torch.from_numpy(pred_joints_[step * batch_size:step * batch_size + curr_batch_size]).to(device)

            # torch.set_printoptions(threshold=10000)
            # print(pred_keypoints_3d-gt_keypoints_3d)

            if keypoint is not None:
                print("kp-wise")
                pred_keypoints_3d = pred_keypoints_3d[:, keypoint:keypoint+1, :]
                gt_keypoints_3d = gt_keypoints_3d[:, keypoint:keypoint+1, :]
            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error (PA-MPJPE)
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Vertices to vertices (PVE)
            if '3dpw' in dataset_name or 'agora' in dataset_name:
                v_error = torch.sqrt(((pred_vertices_aligned - gt_vertices_aligned) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
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
        np.savez(result_file, imgnames=imgnames_, pred_poses=smpl_pose, pred_cams=smpl_camera)
        # res_dict = {'imgnames': imgnames_,
        #             'pred_cams': smpl_camera,
        #             'pred_vertices': pred_vertices_}
        # np.save(result_file, res_dict)
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

    if keypoint or category:
        print("Evaluating on specific splits...")
        with open(os.path.join('subtest_log_{}.txt'.format(out_num)), mode='a', encoding='utf-8') as logger:
            if keypoint is not None:
                keypoint = constants.LSP_JOINT_NAMES[keypoint]
            print('Category: {} - Keypoint {}: {} samples'.format(category, keypoint, len(dataset)), 1000 * mpjpe.mean(), 1000 * recon_err.mean(), file=logger)


    if with_train:
        summary_writer.add_scalar('MPJPE', 1000 * mpjpe.mean(), eval_epoch)
        summary_writer.add_scalar('PA-MPJPE', 1000 * recon_err.mean(), eval_epoch)
        summary_writer.add_scalar('PVE', 1000 * pve.mean(), eval_epoch)
        return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()
    


def run_evaluation_agora(model, dataset_name, dataset, result_file,
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
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
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
        gt_betas = batch['betas'][:, :10].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        gt_verts = batch['gt_vert'].to(device)
        gt_joints = batch['pose_3d'][:, :25]  # 3D pose
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
        # occl = batch['occlusion'].to(device)
        # print(occl)
        # print(gt_keypoints_2d_full)
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
                pred_rotmat, pred_betas, pred_camera, _1 = model(images, bbox_info, n_extraviews=n_views-1)
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

            # num = 0
            # for idx in tqdm(np.random.choice(len(img_names), 4, replace=False)):
            for idx in range(len(img_names)):
                num = step * batch_size + idx
                name = img_names[idx].split('/')[-1].split('.')[0]
                print(name)
                renderer = Renderer(focal_length=focal_length[idx],
                                img_res=[img_w[idx], img_h[idx]], faces=smpl_neutral.faces)
                # print(crop_w, crop_h)
                crop_renderer = Renderer(focal_length=5000,
                                    img_res=[crop_w, crop_h], faces=smpl_neutral.faces)
                image_mat = np.ascontiguousarray((images[idx]*255.0).permute(1, 2, 0).cpu().detach().numpy())
                for kp in gt_keypoints_2d[idx, :29]:
                    cv2.circle(image_mat, (int(kp[0]),int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                
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
                # images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, grid=False)
                images_pred = renderer.visualize_tb(gt_vertices[idx].unsqueeze(0), pred_vertices[idx].unsqueeze(0), pred_cam_full[idx].unsqueeze(0), image_full, grid=False, save_path='eval_result/img_viz/{}_{}_img_gt.obj'.format(dataset_name, num))
                images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images[idx:idx+1], grid=False, save_path='eval_result/img_viz/{}_{}_img_pred.obj'.format(dataset_name, num))
                cv2.imwrite('eval_result/eval_img_viz/{}_{}_img_visualized.jpg'.format(dataset_name, num), np.ascontiguousarray(images_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                cv2.imwrite('eval_result/eval_img_viz/{}_{}_img_crop_visualized.jpg'.format(dataset_name, num), np.ascontiguousarray(images_crop_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
                cv2.imwrite('eval_result/eval_img_viz/{}_{}_img_full_origin.jpg'.format(dataset_name, num), rgb_img[:, :, ::-1])
                cv2.imwrite('eval_result/eval_img_viz/{}_{}_img_crop_origin.jpg'.format(dataset_name, num), image_mat[:, :, ::-1])
                # num = num+1
                

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
            elif dataset_name == 'agora':
                print("Using smpl joints as gt 3d joints!", has_smpl[0])
                # gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                # gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
                torch.set_printoptions(threshold=10000)
                # print(gender)
                # gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_vertices = gt_verts.float()
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                # gt_keypoints_3d = gt_keypoints_3d
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 
                # gt_vertices = gt_verts.float()
                # gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                # gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                # gt_keypoints_3d = gt_keypoints_3d
                # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
                gt_vertices_aligned = gt_vertices - gt_pelvis
                print(gt_keypoints_3d.shape)

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            # print('joints', pred_keypoints_3d.shape)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            # pred_keypoints_3d = pred_keypoints_3d
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_vertices_aligned = pred_vertices - pred_pelvis

            # torch.set_printoptions(threshold=10000)
            # print(pred_keypoints_3d, gt_keypoints_3d)

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error (PA-MPJPE)
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Vertices to vertices (PVE)
            if '3dpw' in dataset_name or 'agora' in dataset_name:
                v_error = torch.sqrt(((pred_vertices_aligned - gt_vertices_aligned) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                pve[step * batch_size:step * batch_size + curr_batch_size] = v_error
            
            with open('eval_result/result.txt', 'a+') as rf:
                print('Step {} --> MPJPE: {}, PA_MPJPE: {}, PVE: {}'.format(step, 1000*error, 1000*r_error, 1000*v_error), file=rf)


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
    args = parser.parse_args()
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
    if not args.use_res:
        model = hmr(config.SMPL_MEAN_PARAMS, name=args.model_name, bbox_type=args.bbox_type, encoder=args.encoder)
        # model = hmr(config.SMPL_MEAN_PARAMS, name=args.model_name)
        checkpoint = torch.load(args.checkpoint)
        # print(checkpoint.keys())
        ## Edit keys in original weights file from CLIFF
        # print(checkpoint.keys())
        # old_dict = checkpoint['state_dict']
        # old_dict = checkpoint['model']
        # new_dict = OrderedDict([(k.replace('module.', ''), v) for k,v in old_dict.items()])
        # new_dict = OrderedDict([(k.replace('model.', ''), v) for k,v in old_dict.items()])
        # new_dict = OrderedDict([(k.replace('head.', ''), v) for k,v in new_dict.items()])
        # new_dict = OrderedDict([(k.replace('encoder.', ''), v) for k,v in new_dict.items()])
        # for k,v in new_dict.items():
        #     print(k, v, file=logger)
        # # model.load_state_dict(new_dict, strict=True)
        # model_dict = {}
        # model_dict['model'] = new_dict
        # torch.save(model_dict, args.checkpoint[:-3]+'_fitted.pt')

        # print('Loading pretrained weights for HRNet backbone...')
        # new_dict = OrderedDict([(k.replace('module.', ''), v) for k,v in old_dict.items()])
        # model.load_state_dict(new_dict, strict=True)
        # model_dict = {}
        # model_dict['model'] = new_dict
        # torch.save(model_dict, args.checkpoint[:-3]+'_fitted.pt')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
    else:
        model = None
    # Setup evaluation dataset
    if args.by_category and not args.by_keypoint:
        for cate in ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting', 'Waiting', 'Walk']:
            dataset = BaseDataset(args, args.dataset, is_train=False, bbox_type=args.bbox_type, category=cate)
            run_evaluation(model, args.dataset, dataset, args.result_file,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            log_freq=args.log_freq,
                            out_num=args.out_num,
                            viz=args.viz,
                            model_aux=None,
                            bbox_type=args.bbox_type,
                            use_extra=args.use_extraviews,
                            use_fuse=args.use_fuse,
                            category=cate
                            )
    elif args.by_keypoint and not args.by_category:
        dataset = BaseDataset(args, args.dataset, is_train=False, bbox_type=args.bbox_type)
        for kp in range(14):
            run_evaluation(model, args.dataset, dataset, args.result_file,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            log_freq=args.log_freq,
                            out_num=args.out_num,
                            viz=args.viz,
                            model_aux=None,
                            bbox_type=args.bbox_type,
                            use_extra=args.use_extraviews,
                            use_fuse=args.use_fuse,
                            category=None,
                            keypoint=kp
                            )
    else:
        dataset = BaseDataset(args, args.dataset, is_train=False, bbox_type=args.bbox_type, val=args.val)
    # dataset = BaseDataset(None, '3dpw', is_train=False, use_extraviews=True)
    # Run evaluation
        if args.dataset == 'agora':
            run_evaluation_agora(model, args.dataset, dataset, args.result_file,
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
            run_evaluation(model, args.dataset, dataset, args.result_file,
                        batch_size=args.batch_size,
                        shuffle=args.shuffle,
                        log_freq=args.log_freq,
                        out_num=args.out_num,
                        viz=args.viz,
                        model_aux=None,
                        bbox_type=args.bbox_type,
                        use_extra=args.use_extraviews,
                        use_fuse=args.use_fuse,
                        use_res=args.use_res,
                        n_views=args.n_views
                        )
