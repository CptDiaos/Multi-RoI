import sys
sys.path.append('../pixelnerf-master')
import torch
import torch.nn as nn
import numpy as np
import os
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
from torchvision.transforms import Normalize

from datasets import BaseDataset, MixedDataset
from models import hmr, SMPL
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, cam_crop2full
from utils.imutils import flip_img
from utils.renderer import Renderer
from utils import BaseTrainer
from eval import run_evaluation
from torchvision.utils import make_grid

import config
import constants
from .fits_dict import FitsDict
import time

class Charbonnier_Ada(nn.Module):
    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss


class Trainer(BaseTrainer):

    def init_fn(self):
        if self.options.train_dataset is not None:
            self.train_ds = BaseDataset(self.options, self.options.train_dataset, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)
        else:
            self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)
        
        self.eval_dataset = BaseDataset(self.options, self.options.eval_dataset, is_train=False, bbox_type=self.options.bbox_type)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True, name='hmr_extraviews_hrnet', bbox_type=self.options.bbox_type).to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.render_focal_length = constants.FOCAL_LENGTH
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.use_pseudo = self.options.use_pseudo
        self.nviews = 5

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
            img_w):
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
            (pred_keypoints2d_full, torch.ones(batch_size, 49, 1).to(device)),
            dim=2)
        # trans @ pred_keypoints2d2
        pred_keypoints2d_bbox = torch.einsum('bij,bkj->bki', crop_trans,
                                        pred_keypoints2d)

        # images = img
        # images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        # images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        # cropped_img = (images[1].permute((1, 2, 0)).cpu().numpy()[..., ::-1] * 255.0).astype('uint8').copy()
        # ori_img = cv2.imread(img_name[1])
        # if is_flipped[1]:
        #     ori_img = np.ascontiguousarray(flip_img(ori_img), dtype=np.uint8)
        # MAR = cv2.getRotationMatrix2D((int(crop_center[1][0]), int(crop_center[1][1])), int(rot[1]), 1.0)
        # rotated_img = cv2.warpAffine(ori_img.copy(), MAR, (int(img_w[1][0]), int(img_h[1][0])))

        # pred_keypoints2d_bbox = torch.zeros_like(pred_keypoints2d_full, device=device)
        # not_flipped = torch.zeros_like(is_flipped, device=device)

        # for kp in gt_keypoints2d_full[1][25:]:
        #     cv2.circle(ori_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
        # for kp in pred_keypoints2d_full[1][25:]:
        #     cv2.circle(rotated_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0), thickness=-1)


        # for b_ind in range(pred_keypoints2d_bbox.shape[0]):
        #     pred_keypoints2d_bbox[b_ind] = j2d_processing_torch(pred_keypoints2d_full[b_ind], crop_center[b_ind],
        #                                                         crop_scale[b_ind], r=0, f=not_flipped[b_ind],
        #                                                         crop_t=trans[b_ind])


        # for kp in gt_keypoints2d[1][25:]:
        #     cv2.circle(cropped_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
        # for kp in pred_keypoints2d_bbox[1][25:]:
        #     cv2.circle(cropped_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0), thickness=-1)
        # cv2.imshow('cropped_img', cropped_img)
        # cv2.imshow('rotated_img', rotated_img)
        # cv2.imshow('ori_img', ori_img)
        # cv2.waitKey()

        pred_keypoints2d_bbox[:, :, :2] = 2. * pred_keypoints2d_bbox[:, :, :2] / constants.IMG_RES - 1.
        gt_keypoints2d[:, :, :2] = 2. * gt_keypoints2d[:, :, :2] / constants.IMG_RES - 1.


        loss = self.keypoint_loss(pred_keypoints2d_bbox.float(), gt_keypoints2d.float(),
                                  self.options.openpose_train_weight,
                                  self.options.gt_train_weight)

        return loss

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch, cur_epoch, cur_step):
        self.model.train()

        # Get data from the batch
        images = input_batch['img']  # input image
        img_name = input_batch['imgname']
        gt_keypoints_2d_full = input_batch['keypoints_full']
        gt_keypoints_2d = input_batch['keypoints']  # 2D keypoints
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d']  # 3D pose
        has_smpl = input_batch['has_smpl'].byte()  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte()  # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        crop_trans = input_batch['crop_trans']
        full_trans = input_batch['full_trans']
        inv_trans = input_batch['inv_trans']
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        indices = input_batch['sample_index']  # index of example inside its dataset
        batch_size = images.shape[0]
        bbox_info = input_batch['bbox_info']
        center, scale, focal_length = input_batch['center'], input_batch['scale'], input_batch['focal_length'].float()
        if self.options.use_extraviews:
            bboxes_info_extra = input_batch['bboxes_info_extra']
            imgs_extra = input_batch['imgs_extra']
        


        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # for b_ind in range(gt_keypoints_2d_full.shape[0]):
        #     gt_keypoints_2d_orig[b_ind] = j2d_processing_torch(gt_keypoints_2d_full[b_ind], center[b_ind], scale[b_ind], r=0, f=is_flipped[b_ind], crop_t=crop_tran[b_ind])
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * constants.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        # gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig,
        #                                 focal_length=np.array(focal_length.detach().cpu(), dtype=int),
        #                                 img_size=self.options.img_res)

        # Feed images in the network to predict camera and SMPL parameters
        # pred_rotmat, pred_betas, pred_camera = self.model(images, bbox_info)
        if self.options.use_extraviews:
            images = torch.cat([images, imgs_extra], 1)
            # images = images.repeat(1, 5, 1, 1)
            bbox_info = torch.cat([bbox_info, bboxes_info_extra], 1)

        start = time.time()
        pred_rotmat, pred_betas, pred_camera = self.model(images, bbox_info)

        end = time.time()

        # print("Model iteration time:", end-start)


        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_crop = torch.stack([pred_camera[:, 1],
                                     pred_camera[:, 2],
                                     2 * self.render_focal_length / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                    dim=-1)

        img_h, img_w = input_batch['img_h'].view(-1, 1), input_batch['img_w'].view(-1, 1)

        full_img_shape = torch.hstack((img_h, img_w))
        pred_cam_full = cam_crop2full(pred_camera, center, scale, full_img_shape,
                                      focal_length).to(torch.float32)
        camera_center_bbox = torch.zeros(batch_size, 2)
        camera_center = torch.hstack((img_w, img_h)) / 2

        # Compute loss on SMPL parameters
        # loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl)

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
            img_w
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        # loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape + \
               self.options.keypoint_loss_weight * loss_keypoints + \
               self.options.keypoint_loss_weight * loss_keypoints_3d + \
               self.options.pose_loss_weight * loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas
        loss *= 60

        # Do backprop
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
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}
        print(losses)
        return output, losses
    
    def test(self, epoch):
        self.model.eval()
        mpjpe, pa_mpjpe = run_evaluation(self.model, '3dpw', self.eval_dataset, None,
                            batch_size=100,
                            shuffle=False,
                            log_freq=50, 
                            with_train=True, eval_epoch=epoch, summary_writer=self.summary_writer, use_extra=self.options.use_extraviews, n_views=self.nviews)
        results = {'mpjpe': mpjpe,
                   'pa_mpjpe': pa_mpjpe}
        return results

    def train_summaries(self, input_batch, output, losses):

        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        if self.options.use_extraviews and self.nviews>1:
            images_extra = input_batch['imgs_extra']
            images_extra = images_extra * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1).repeat(1, self.nviews-1, 1, 1)
            images_extra = images_extra + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1).repeat(1, self.nviews-1, 1, 1)

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
        img_to_sum = np.random.choice(len(img_names), 4, replace=False)
        num = 0
        for idx in img_to_sum:
            crop_renderer = Renderer(focal_length=5000,
                                        img_res=[224, 224], faces=self.smpl.faces)
            renderer = Renderer(focal_length=focal_length[idx],
                                img_res=[img_w[idx], img_h[idx]], faces=self.smpl.faces)
            rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
            if is_flipped[idx]:
                rgb_img = flip_img(rgb_img)
            MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
            rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))), (2, 0, 1)) / 255.0
            image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
            images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_t[idx].unsqueeze(0),
                                                image_full)
            # images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0), pred_cam_crop[idx].unsqueeze(0), images[idx:idx+1])
            self.summary_writer.add_image('pred_mesh_{}_{}'.format(dataset[idx], num), images_pred, self.step_count)
            # self.summary_writer.add_image('pred_crop_mesh_{}_{}'.format(dataset[idx], num), images_crop_pred, self.step_count)
            if self.options.use_extraviews and self.nviews>1:
                images_extra_list = []
                for i in range(self.nviews-1):
                    images_extra_list.append(images_extra[idx, 3*i:3*(i+1)])
                images_extra_list = make_grid(images_extra_list, nrow=4)
                self.summary_writer.add_image('images_extra_{}_{}'.format(dataset[idx], num), images_extra_list, self.step_count)
            num += 1

        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
        if self.use_pseudo:
            err_s = output['err_s']
            self.summary_writer.add_scalar('S', err_s.item(), self.step_count)
