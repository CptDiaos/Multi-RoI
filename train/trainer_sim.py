import torch
import torch.nn as nn
import numpy as np
import os
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
from torchvision.transforms import Normalize
from models.conloss import SupConLoss
import matplotlib.pyplot as plt

from datasets import BaseDataset, MixedDataset, MixedDataset_wo3dpw
from models import hmr, SMPL, SMPL_agora
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, cam_crop2full
from utils.imutils import flip_img
from utils.renderer import Renderer
from utils import BaseTrainer
from eval import run_evaluation, run_evaluation_agora
from torchvision.utils import make_grid

import config
import constants
from .fits_dict import FitsDict
import time




class Trainer(BaseTrainer):

    def init_fn(self):
        if self.options.train_dataset is not None:
            self.train_ds = BaseDataset(self.options, self.options.train_dataset, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)
        else:
            self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)
        
        self.eval_dataset = BaseDataset(self.options, self.options.eval_dataset, is_train=False, bbox_type=self.options.bbox_type)

        if self.options.use_msa:
            model_name = 'hmr_msa'
        else:
            model_name = 'hmr_sim'
        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True, name=model_name, n_extraviews=self.options.n_views-1, bbox_type=self.options.bbox_type, wo_enc=self.options.wo_enc, wo_fuse=self.options.wo_fuse, encoder=self.options.encoder).to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        if self.options.train_dataset == 'old agora':
            print('Training on AGORA dataset..')
            self.joints_idx = 0
            self.joints_num = 24
            self.smpl = SMPL_agora(config.SMPL_MODEL_DIR,
                            batch_size=self.options.batch_size,
                            create_transl=False).to(self.device)
        else:
            self.joints_idx = 25
            self.joints_num = 49
            # self.smpl = SMPL(config.SMPL_MODEL_DIR,
            #                     batch_size=self.options.batch_size,
            #                     create_transl=False).to(self.device)
            self.smpl = SMPL(config.SMPL_MODEL_DIR,
                                batch_size=self.options.batch_size,
                                create_transl=False).to(self.device)
            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     batch_size=self.options.batch_size,
                            create_transl=False).to(self.device)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       batch_size=self.options.batch_size,
                            create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_regularize = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.render_focal_length = constants.FOCAL_LENGTH
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.use_pseudo = self.options.use_pseudo
        self.n_views = self.options.n_views
        self.conloss = SupConLoss()
        self.bbox_type = self.options.bbox_type
        if self.bbox_type=='square':
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type=='rect':
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)


    def compute_keypoints2d_loss_cliff(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            camera_center: torch.Tensor,
            focal_length: torch.Tensor,
            crop_trans,
            img_h,
            img_w,
            img,
            dataset,
            img_name,
            is_flipped,
            crop_center,
            gt_keypoints2d_full,
            rot,
            viz,
            dbg_dataset,
            agora_occ=None):
        """Compute loss for 2d keypoints."""
        img_shape = torch.cat((img_w, img_h), dim=1)
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :].float()

        device = gt_keypoints2d.device
        batch_size = pred_keypoints3d.shape[0]

        pred_keypoints2d_full = perspective_projection(
            pred_keypoints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(
                batch_size, -1, -1),
            translation=pred_cam,
            focal_length=focal_length,
            camera_center=camera_center)
        

        img_size = torch.max(img_shape, dim=1)[0]

        pred_keypoints2d = torch.cat(
            (pred_keypoints2d_full, torch.ones(batch_size, self.joints_num, 1).to(device)),
            dim=2)
        # trans @ pred_keypoints2d2
        pred_keypoints2d_bbox = torch.einsum('bij,bkj->bki', crop_trans,
                                        pred_keypoints2d)
        if viz:
            idx = dataset.index(dbg_dataset)
            print('Viusalizing ', idx)
            for view in range(self.n_views):
                # print(dataset)
                # print(pred_keypoints2d_full[idx])
                images = img[:, view*3: (view+1)*3]
                images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                cropped_img = (images[idx].permute((1, 2, 0)).cpu().numpy()[..., ::-1] * 255.0).astype('uint8').copy()
                ori_img = cv2.imread(img_name[idx])
                if dbg_dataset=='3dpw':
                    ori_img = cv2.resize(ori_img, (ori_img.shape[1]//2, ori_img.shape[0]//2))  # when 3dpw
                if is_flipped[idx]:
                    ori_img = np.ascontiguousarray(flip_img(ori_img), dtype=np.uint8)

                for kp in gt_keypoints2d_full[idx][self.joints_idx:]:
                    cv2.circle(ori_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0), thickness=5)

                MAR = cv2.getRotationMatrix2D((int(crop_center[idx][0]), int(crop_center[idx][1])), int(rot[idx]), 1.0)
                rotated_img = cv2.warpAffine(ori_img.copy(), MAR, (int(img_w[idx][0]), int(img_h[idx][0])))

                # pred_keypoints2d_bbox = torch.zeros_like(pred_keypoints2d_full, device=device)
                # not_flipped = torch.zeros_like(is_flipped, device=device)

                for kp in pred_keypoints2d_full[idx*self.n_views+view][self.joints_idx:]:
                    cv2.circle(rotated_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                
                # print(pred_keypoints2d_full[idx])

                
                for kp in gt_keypoints2d[idx*self.n_views+view][self.joints_idx:]:
                    cv2.circle(cropped_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0), thickness=-1)
                for kp in pred_keypoints2d_bbox[idx*self.n_views+view][self.joints_idx:]:
                    cv2.circle(cropped_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imwrite('eval_result/img_viz/train_img_crop_visualized_{}.jpg'.format(view), np.ascontiguousarray(cropped_img))
                cv2.imwrite('eval_result/img_viz/train_img_full_visualized_{}.jpg'.format(view), np.ascontiguousarray(rotated_img))
                # cv2.imshow('cropped_img', cropped_img)
                # cv2.imshow('rotated_img', rotated_img)
                # cv2.imshow('ori_img', ori_img)
                # cv2.waitKey()

        # pred_keypoints2d_bbox[:, :, :2] = 2. * pred_keypoints2d_bbox[:, :, :2] / constants.IMG_RES - 1.
        # gt_keypoints2d[:, :, :2] = 2. * gt_keypoints2d[:, :, :2] / constants.IMG_RES - 1.
        pred_keypoints2d_bbox[:, :, 0] = 2. * pred_keypoints2d_bbox[:, :, 0] / self.crop_w - 1.
        gt_keypoints2d[:, :, 0] = 2. * gt_keypoints2d[:, :, 0] / self.crop_w - 1.
        pred_keypoints2d_bbox[:, :, 1] = 2. * pred_keypoints2d_bbox[:, :, 1] / self.crop_h - 1.
        gt_keypoints2d[:, :, 1] = 2. * gt_keypoints2d[:, :, 1] / self.crop_h - 1.


        loss = self.keypoint_loss(pred_keypoints2d_bbox.float(), gt_keypoints2d.float(),
                                  self.options.openpose_train_weight,
                                  self.options.gt_train_weight, agora_occ=agora_occ)

        return loss

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight, agora_occ = None):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        # torch.set_printoptions(threshold=100000)
        # print(conf)
        
        if agora_occ is not None:
            kp2d_vis_w = agora_occ.view(-1, 1, 1).repeat(1, pred_keypoints_2d.shape[1], pred_keypoints_2d.shape[2])
        else:
            conf[:, :25] *= openpose_weight
            conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(kp2d_vis_w*pred_keypoints_2d, kp2d_vis_w*gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, dataset=None, viz=False, dbg_dataset='3dpw', use_model=False, agora_occ = None):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        # pred_keypoints_3d = pred_keypoints_3d[:, self.joints_idx:, :]
        # print(pred_keypoints_3d.shape, gt_keypoints_3d.shape)
        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        if use_model:
            conf = torch.ones_like(gt_keypoints_3d[:, :, -1].unsqueeze(-1), device=self.device)
            gt_keypoints_3d = gt_keypoints_3d.clone()
        else:
            conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
            gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()

        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]

        if agora_occ is not None:
            kp3d_vis_w = agora_occ.view(-1, 1, 1).repeat(1, pred_keypoints_3d.shape[1], pred_keypoints_3d.shape[2])

        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            if viz:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                idx = dataset.index(dbg_dataset)
                p_points = pred_keypoints_3d[idx, :, :3].cpu().detach().numpy()
                g_points = gt_keypoints_3d[idx, :, :3].cpu().detach().numpy()
                # for joint_idx in range(pred_keypoints_3d.shape[1]):
                ax.scatter3D(p_points[:, 0], p_points[:, 1], p_points[:, 2], c='red')
                ax.scatter3D(g_points[:, 0], g_points[:, 1], g_points[:, 2], c='green')
                    # ax.scatter3D(x=g_points[joint_idx, 0], y=g_points[joint_idx, 1], z=g_points[joint_idx, 2], c='red')
                plt.savefig('eval_result/img_viz/3D_joints.jpg')
                plt.close()
            return (conf * self.criterion_keypoints(kp3d_vis_w*pred_keypoints_3d, kp3d_vis_w*gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)
        

    def shape_loss(self, pred_vertices, gt_vertices, valid_vert, dataset=None, viz=False, dbg_dataset='3dpw', agora_occ = None):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[valid_vert == 1]
        gt_vertices_with_shape = gt_vertices[valid_vert == 1]
        J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        if agora_occ is not None:
            print()
            vert_vis_w = agora_occ.view(-1, 1, 1).repeat(1, pred_vertices_with_shape.shape[1], pred_vertices_with_shape.shape[2])[valid_vert == 1]
        J_regressor_batch_gt = J_regressor[None, :].expand(gt_vertices_with_shape.shape[0], -1, -1).to(self.device)
        target_keypoints_3d = torch.matmul(J_regressor_batch_gt, gt_vertices_with_shape)
        target_pelvis = target_keypoints_3d[:, [0],:].clone()
        target_vertices_aligned = gt_vertices_with_shape - target_pelvis

        est_keypoints_3d = torch.matmul(J_regressor_batch_gt, pred_vertices_with_shape)
        est_pelvis = est_keypoints_3d[:, [0],:].clone()
        est_vertices_aligned = pred_vertices_with_shape - est_pelvis

        if viz and len(gt_vertices_with_shape)>0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            idx = dataset.index(dbg_dataset)
            p_points = est_vertices_aligned[idx, :, :3].cpu().detach().numpy()
            g_points = target_vertices_aligned[idx, :, :3].cpu().detach().numpy()
            # for joint_idx in range(pred_keypoints_3d.shape[1]):
            ax.scatter3D(p_points[:, 0], p_points[:, 1], p_points[:, 2], c='red')
            ax.scatter3D(g_points[:, 0], g_points[:, 1], g_points[:, 2], c='green')
                # ax.scatter3D(x=g_points[joint_idx, 0], y=g_points[joint_idx, 1], z=g_points[joint_idx, 2], c='red')
            plt.savefig('eval_result/img_viz/3D_vertices.jpg')
            plt.close()
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(vert_vis_w*est_vertices_aligned, vert_vis_w*target_vertices_aligned)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl, valid_beta, agora_occ = None):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[valid_beta == 1]
        gt_betas_valid = gt_betas[valid_beta == 1]
        if agora_occ is not None:
            pose_vis_w = agora_occ.view(-1, 1, 1, 1).repeat(1, 24, 3, 3)[has_smpl == 1]
            beta_vis_w = agora_occ.view(-1, 1).repeat(1, 10)[valid_beta == 1]
        if len(pred_rotmat_valid) > 0:
            # print(pred_rotmat_valid.shape, gt_rotmat_valid.shape)
            loss_regr_pose = self.criterion_regr(pose_vis_w*pred_rotmat_valid, pose_vis_w*gt_rotmat_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
        if len(pred_betas_valid) > 0:
            loss_regr_betas = self.criterion_regr(beta_vis_w*pred_betas_valid, beta_vis_w*gt_betas_valid)
        else:
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)

        return loss_regr_pose, loss_regr_betas
    
    def camera_losses(self, crop_cam, center, scale, full_img_shape, agora_occ = None):
        ori_b = center.shape[0]//self.n_views
        img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
        cx, cy, b = center[:, 0], center[:, 1], scale * 200.
        w_2, h_2 = img_w / 2., img_h / 2.
        bs = b * crop_cam[:, 0] + 1e-9
        main_ind = torch.full((center.shape[0],), 0) + self.n_views*torch.arange(ori_b).unsqueeze(1).repeat(1, self.n_views).view(-1,)
        views_ind = torch.arange(self.n_views).unsqueeze(0).repeat(ori_b, 1).view(-1,) + self.n_views*torch.arange(ori_b).unsqueeze(1).repeat(1, self.n_views).view(-1,)
        # print(main_ind, views_ind)
        # reg_x = crop_cam[main_ind, 1] - crop_cam[views_ind, 1] - (2 * ((cx[views_ind] - w_2[views_ind])-(cx[main_ind] - w_2[main_ind])) / bs)
        # reg_y = crop_cam[main_ind, 2] - crop_cam[views_ind, 2] - (2 * ((cy[views_ind] - h_2[views_ind])-(cy[main_ind] - h_2[main_ind])) / bs)
        reg_x = crop_cam[main_ind, 1] - crop_cam[views_ind, 1] - (2 * ((cx[views_ind] - w_2[views_ind])/bs[views_ind]-(cx[main_ind] - w_2[main_ind])/bs[main_ind]))
        reg_y = crop_cam[main_ind, 2] - crop_cam[views_ind, 2] - (2 * ((cy[views_ind] - h_2[views_ind])/bs[views_ind]-(cy[main_ind] - h_2[main_ind])/bs[main_ind]))
        reg_s = (bs[views_ind] - bs[main_ind])* 1e-4
        # print(reg_x.shape, reg_y.shape, reg_s.shape)
        # print((reg_x+reg_y).mean(), reg_s.mean())
        if agora_occ is not None:
            vis_w = agora_occ
        return self.criterion_regularize(vis_w*(reg_x+reg_y), torch.zeros_like(reg_x, device=reg_x.device)) + \
                self.criterion_regularize(vis_w*reg_s, torch.zeros_like(reg_s, device=reg_x.device))


    def train_step(self, input_batch, cur_epoch, cur_step):
        if not self.options.viz_debug:
            self.model.train()
        else:
            self.model.eval()
        n_views = self.n_views

        # Get data from the batch
        images = input_batch['img']  # input image
        batch_size = images.shape[0]
        img_name = input_batch['imgname']
        gt_keypoints_2d_full = input_batch['keypoints_full'][:, :self.joints_num]
        gt_keypoints_2d = input_batch['keypoints'][:, :self.joints_num]  # 2D keypoints
        gender = input_batch['gender'].unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,)
        gt_pose = input_batch['pose'].unsqueeze(1).repeat(1, n_views, 1).view(batch_size*n_views, -1)  # SMPL pose parameters
        gt_betas = input_batch['betas'].unsqueeze(1).repeat(1, n_views, 1).view(batch_size*n_views, -1)  # SMPL beta parameters
        # gt_joints = input_batch['pose_3d'].unsqueeze(1).repeat(1, n_views, 1, 1).view(batch_size*n_views, self.joints_num-self.joints_idx, 4)  # 3D pose
        if self.options.train_dataset == 'agora':
            gt_joints = input_batch['pose_3d'].unsqueeze(1).repeat(1, n_views, 1, 1).view(batch_size*n_views, -1, 4)[:, :self.joints_num]  # 3D pose
            # isValid = input_batch['isValid']
            # print(input_batch['occlusion'])
            # occls_rate = (1.-(input_batch['occlusion']/100.)+1e-2).unsqueeze(1).repeat(1, n_views).view(batch_size*n_views)
            occls_rate = torch.ones((batch_size*n_views), device=self.device)
            # gt_vertices = input_batch['gt_vert'].unsqueeze(1).repeat(1, n_views, 1, 1).view(batch_size*n_views, -1, 3).float()
            # print(occls_rate)
        else:
            occls_rate = torch.ones((batch_size*n_views), device=self.device)
            gt_joints = input_batch['pose_3d'].unsqueeze(1).repeat(1, n_views, 1, 1).view(batch_size*n_views, self.joints_num-self.joints_idx, 4)  # 3D pose
        has_smpl = input_batch['has_smpl'].unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,).byte()  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,).byte()  # flag that indicates whether 3D pose is valid
        valid_vert = input_batch['valid_vert'].byte().unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,).byte()
        valid_beta = input_batch['valid_beta'].byte().unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,).byte()
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        crop_trans = input_batch['crop_trans']
        full_trans = input_batch['full_trans']
        inv_trans = input_batch['inv_trans']
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        indices = input_batch['sample_index']  # index of example inside its dataset
        bbox_info = input_batch['bbox_info']
        center, scale, focal_length = input_batch['center'], \
                                        input_batch['scale'], \
                                            input_batch['focal_length'].unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,).float()
                                        # input_batch['scale'].unsqueeze(1).repeat(1, n_views).view(batch_size*n_views,), \
        centers_extra = input_batch['centers_extra']
        scales_extra = input_batch['scales_extra']
        # center, scale, focal_length = input_batch['center'], \
        #                                 input_batch['scale'], \
        #                                     input_batch['focal_length'].float()

        bboxes_info_extra = input_batch['bboxes_info_extra']
        imgs_extra = input_batch['imgs_extra']
        gt_keypoints_2d_extra = input_batch['keypoints_extra'][:, :, :self.joints_num]
        crop_trans_extra = input_batch['crop_trans_extra']
        # print(gt_keypoints_2d.shape, gt_pose.shape, gt_betas.shape, gt_joints.shape,has_smpl.shape,has_pose_3d.shape)
        # print(center.shape, centers_extra.shape, scale.shape, focal_length.shape)

        gt_keypoints_2d = torch.cat([gt_keypoints_2d.unsqueeze(1), gt_keypoints_2d_extra], 1).view(batch_size*n_views, self.joints_num, 3)
        crop_trans = torch.cat([crop_trans.unsqueeze(1), crop_trans_extra], 1).view(batch_size*n_views, 2, 3)
        center = torch.cat([center.unsqueeze(1), centers_extra], 1).view(batch_size*n_views, 2)
        scale = torch.cat([scale.unsqueeze(1), scales_extra], 1).view(batch_size*n_views,)
        
        # print(gt_keypoints_2d.shape, gt_pose.shape, gt_betas.shape, gt_joints.shape,has_smpl.shape,has_pose_3d.shape)
        # print(center.shape, scale.shape, focal_length.shape, crop_trans.shape)


        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        # gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices
        # gt_vertices = self.smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas[:, :10]).vertices 
        # gt_vertices_female = self.smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas[:, :10]).vertices 
        # # print(gender)
        # gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # for b_ind in range(gt_keypoints_2d_full.shape[0]):
        #     gt_keypoints_2d_orig[b_ind] = j2d_processing_torch(gt_keypoints_2d_full[b_ind], center[b_ind], scale[b_ind], r=0, f=is_flipped[b_ind], crop_t=crop_tran[b_ind])
        # gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        gt_keypoints_2d_orig[:, :, 0] = 0.5 * self.crop_w * (gt_keypoints_2d_orig[:, :, 0] + 1)
        gt_keypoints_2d_orig[:, :, 1] = 0.5 * self.crop_h * (gt_keypoints_2d_orig[:, :, 1] + 1)

        # Feed images in the network to predict camera and SMPL parameters
        # pred_rotmat, pred_betas, pred_camera = self.model(images, bbox_info)
        if self.options.use_extraviews:
            images = torch.cat([images, imgs_extra], 1)
            # images = images.repeat(1, 5, 1, 1)
            bbox_info = torch.cat([bbox_info, bboxes_info_extra], 1)
            shifts_extra = input_batch['shifts_extra']
            rescales_extra = input_batch['rescales_extra']

        start = time.time()
        pred_rotmat, pred_betas, pred_camera, xf_g = self.model(images, bbox_info, n_extraviews=n_views-1)
        # pred_rotmat, pred_rotmat_extra= pred_rotmat_all.view(batch_size, n_views, 24, 3, 3)[:, 0], pred_rotmat_all.view(batch_size, n_views, 24, 3, 3)[:, 1:]
        # pred_betas, pred_betas_extra= pred_betas_all.view(batch_size, n_views, -1)[:, 0], pred_betas_all.view(batch_size, n_views, -1)[:, 1:]
        # pred_camera, pred_camera_extra= pred_camera_all.view(batch_size, n_views, -1)[:, 0], pred_camera_all.view(batch_size, n_views, -1)[:, 1:]
        # print(pred_rotmat.shape, pred_betas.shape, pred_camera.shape)

        end = time.time()
        # print(pred_camera[:5])

        # print("Model iteration time:", end-start)

        label_mask = torch.eye(batch_size, batch_size, device=self.device).unsqueeze(-1).repeat(
            1, self.n_views, self.n_views).view(batch_size*self.n_views, batch_size*self.n_views)
        # print(label_mask)
        
        loss_con = self.conloss(xf_g, label_mask)
        # print(loss_con)


        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints[:, :self.joints_num]
        # pred_vertices = gt_out.vertices
        # pred_joints = gt_out.joints[:, :self.joints_num]

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_crop = torch.stack([pred_camera[:, 1],
                                     pred_camera[:, 2],
                                     2 * self.render_focal_length / (self.bbox_size * pred_camera[:, 0] + 1e-9)],
                                    dim=-1)

        img_h, img_w = input_batch['img_h'].unsqueeze(1).repeat(1, n_views).view(-1, 1), input_batch['img_w'].unsqueeze(1).repeat(1, n_views).view(-1, 1)

        full_img_shape = torch.hstack((img_h, img_w))
        pred_cam_full = cam_crop2full(pred_camera, center, scale, full_img_shape,
                                      focal_length).to(torch.float32)
        camera_center_bbox = torch.zeros(batch_size, 2)
        camera_center = torch.hstack((img_w, img_h)) / 2

        # Compute loss on SMPL parameters
        # loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas[:, :10], has_smpl, valid_beta=valid_beta, agora_occ=occls_rate)

        # Compute 2D reprojection loss for the keypoints
        # loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
        #                                     self.options.openpose_train_weight,
        #                                     self.options.gt_train_weight)
        loss_keypoints = self.compute_keypoints2d_loss_cliff(
            pred_joints,
            pred_cam_full,
            gt_keypoints_2d,
            camera_center,
            focal_length,
            crop_trans,
            img_h,
            img_w,
            img=images,
            dataset=dataset_name,
            img_name=img_name,
            is_flipped=is_flipped,
            crop_center=center,
            gt_keypoints2d_full=gt_keypoints_2d_full,
            rot=rot_angle,
            viz=self.options.viz_debug,
            dbg_dataset=self.options.debug_dataset,
            agora_occ=occls_rate
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints[:, self.joints_idx:], gt_joints, has_pose_3d, dataset=dataset_name, viz=self.options.viz_debug, dbg_dataset=self.options.debug_dataset, agora_occ=occls_rate)

        # Per-vertex loss for the shape
        # loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, valid_vert=valid_vert, dataset=dataset_name, viz=self.options.viz_debug, dbg_dataset=self.options.debug_dataset, agora_occ=occls_rate)

        loss_cam = self.camera_losses(pred_camera, center, scale, full_img_shape, agora_occ=occls_rate)
        # print(loss_cam.shape, loss_cam)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.keypoint_loss_weight * loss_keypoints + \
               self.options.keypoint_loss_weight * loss_keypoints_3d + \
               self.options.shape_loss_weight * loss_shape + \
               self.options.pose_loss_weight * loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas + self.options.cam_loss_weight * loss_cam + self.options.con_loss_weight*loss_con
        loss *= 60
        

        # Do backprop
        if not self.options.viz_debug:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_camera': pred_camera.detach(),
                  'pred_cam_crop': pred_cam_crop.detach(),
                  'pred_cam_full': pred_cam_full.detach(),
                  'dataset': dataset_name,
                  'img_name': img_name,
                  'gt_vertices': gt_vertices}
        losses = {'loss': loss.detach().item(),
                  'loss_cam': loss_cam.detach().item(),
                  'loss_con': loss_con.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}
        # print(losses)
        if self.options.viz_debug:
            print(losses)
        else:
            if cur_step%10==0:
                print(losses)
        return output, losses
    
    def test(self, epoch):
        self.model.eval()
        if self.options.eval_dataset != 'agora':
            mpjpe, pa_mpjpe, pve = run_evaluation(self.model, self.options.eval_dataset, self.eval_dataset, None,
                                batch_size=100,
                                shuffle=False,
                                log_freq=50, 
                                with_train=True, eval_epoch=epoch, summary_writer=self.summary_writer, out_num=4, use_extra=self.options.use_extraviews, use_fuse=True, n_views = self.n_views, bbox_type=self.bbox_type)
        else:
            mpjpe, pa_mpjpe, pve = run_evaluation_agora(self.model, self.options.eval_dataset, self.eval_dataset, None,
                                batch_size=100,
                                shuffle=False,
                                log_freq=8, 
                                with_train=True, eval_epoch=epoch, summary_writer=self.summary_writer, out_num=4, use_extra=self.options.use_extraviews, use_fuse=True, n_views = self.n_views, bbox_type=self.bbox_type, viz=False)
        results = {'mpjpe': mpjpe,
                   'pa_mpjpe': pa_mpjpe,
                   'pve': pve}
        return results

    def train_summaries(self, input_batch, output, losses):

        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        if self.options.use_extraviews and self.n_views>1:
            images_extra = input_batch['imgs_extra']
            images_extra = images_extra * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1).repeat(1, self.n_views-1, 1, 1)
            images_extra = images_extra + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1).repeat(1, self.n_views-1, 1, 1)

        pred_vertices = output['pred_vertices']
        gt_vertices = output['gt_vertices']
        pred_cam_t = output['pred_cam_full']
        pred_cam_crop = output['pred_cam_crop']
        img_names = output['img_name']
        dataset = output['dataset']
        focal_length = input_batch['focal_length']
        center = input_batch['center']
        rot = input_batch['rot_angle']
        is_flipped = input_batch['is_flipped']
        img_h, img_w = input_batch['img_h'], input_batch['img_w']
        if len(img_names)>1:
            img_to_sum = np.random.choice(len(img_names), 4, replace=False)
        else:
            img_to_sum = [0]
        num = 0
        for idx in img_to_sum:
            idx_ext = idx
            if self.options.use_extraviews and self.n_views>1:
                idx_ext = idx*self.n_views
            crop_renderer = Renderer(focal_length=5000,
                                        img_res=[self.crop_w, self.crop_h], faces=self.smpl.faces)
            renderer = Renderer(focal_length=focal_length[idx],
                                img_res=[img_w[idx], img_h[idx]], faces=self.smpl.faces)
            rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
            if dataset[idx]=='3dpw':
                rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2))
            if is_flipped[idx]:
                rgb_img = flip_img(rgb_img)
            MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
            rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))), (2, 0, 1)) / 255.0
            image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
            images_pred = renderer.visualize_tb(pred_vertices[idx_ext].unsqueeze(0), gt_vertices[idx_ext].unsqueeze(0), pred_cam_t[idx_ext].unsqueeze(0),
                                                image_full)
            
            images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx_ext].unsqueeze(0), gt_vertices[idx_ext].unsqueeze(0), pred_cam_crop[idx_ext].unsqueeze(0), images[idx:idx+1])
            if self.options.use_extraviews and self.n_views>1:
                images_extra_list = []
                for i in range(self.n_views-1):
                    image_extra = images_extra[idx:idx+1, 3*i:3*(i+1)]
                    print('image_extra',image_extra.shape)
                    images_crop_aug_pred = crop_renderer.visualize_tb(pred_vertices[idx*self.n_views+i+1].unsqueeze(0), gt_vertices[idx*self.n_views+i+1].unsqueeze(0), pred_cam_crop[idx*self.n_views+i+1].unsqueeze(0), image_extra)
                    self.summary_writer.add_image('pred_crop_mesh_{}_{}_{}'.format(dataset[idx], num, i), images_crop_aug_pred, self.step_count)
            self.summary_writer.add_image('pred_mesh_{}_{}'.format(dataset[idx], num), images_pred, self.step_count)
            self.summary_writer.add_image('pred_crop_mesh_{}_{}'.format(dataset[idx], num), images_crop_pred, self.step_count)

            # crop_renderer = Renderer(focal_length=5000,
            #                             img_res=[self.crop_w, self.crop_h], faces=self.smpl.faces)
            # renderer = Renderer(focal_length=focal_length[idx],
            #                     img_res=[img_w[idx], img_h[idx]], faces=self.smpl.faces)
            # rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
            # if is_flipped[idx]:
            #     rgb_img = flip_img(rgb_img)
            # MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
            # rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))), (2, 0, 1)) / 255.0
            # image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
            # images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_t[idx].unsqueeze(0),
            #                                     image_full)
            # # images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images[idx:idx+1])
            # self.summary_writer.add_image('pred_mesh_{}_{}'.format(dataset[idx], num), images_pred, self.step_count)
            # # self.summary_writer.add_image('pred_crop_mesh_{}_{}'.format(dataset[idx], num), images_crop_pred, self.step_count)
            # if self.options.use_extraviews:
            #     images_extra_list = []
            #     for i in range(4):
            #         images_extra_list.append(images_extra[idx, 3*i:3*(i+1)])
            #     images_extra_list = make_grid(images_extra_list, nrow=4)
            #     self.summary_writer.add_image('images_extra_{}_{}'.format(dataset[idx], num), images_extra_list, self.step_count)
            num += 1

        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
        if self.use_pseudo:
            err_s = output['err_s']
            self.summary_writer.add_scalar('S', err_s.item(), self.step_count)
