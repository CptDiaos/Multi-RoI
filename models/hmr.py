import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from .vit import ViT
from utils.geometry import rot6d_to_rotmat
# from models.gcn import GCN, RelGCN
from itertools import combinations, combinations_with_replacement
from torch.nn.parameter import Parameter
# from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# from .locallyconnected2D import LocallyConnected2d
from .hrnet_re import hrnet_w48, hrnet_w32
from models.backbones.hrnet.cls_hrnet import HighResolutionNet
from models.backbones.hrnet.hrnet_config import cfg
from models.backbones.hrnet.hrnet_config import update_config
import time
from .code import PositionalEncoding
from collections import OrderedDict



class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, bbox_type='square', encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.regression = [self.fc1, self.drop1, self.fc2, self.drop2, self.decpose, self.decshape, self.deccam]
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        print()
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam

class HMR_hrnet(nn.Module):
    """ SMPL Iterative Regressor with HRNet backbone
    """

    def __init__(self, smpl_mean_params, bbox_type='rect', encoder='hr32'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_hrnet, self).__init__()
        npose = 24 * 6
        nbbox = 3
        if encoder=='hr32':
            self.backbone_out = 480
        elif encoder =='hr48':
            self.backbone_out = 720
        else:
            raise Exception("Unsupported backbone!")
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.backbone_out + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.regression = [self.fc1, self.drop1, self.fc2, self.drop2, self.decpose, self.decshape, self.deccam]
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if encoder=='hr32':
            self.backbone=hrnet_w32(pretrained_ckpt_path='logs/pretrained_hrnet/pose_hrnet_w32_256x192.pth', downsample=True, use_conv=True)
        elif encoder=='hr48':
            self.backbone=hrnet_w48(pretrained_ckpt_path='logs/pretrained_hrnet/pose_hrnet_w48_256x192.pth', downsample=True, use_conv=True)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.backbone(x)
        xf = self.avgpool(xf)
        # print(xf.shape)
        xf = xf.view(xf.size(0), -1)
        

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        print()
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam
    

class HMR_vit(nn.Module):
    """ SMPL Iterative Regressor with HRNet backbone
    """

    def __init__(self, smpl_mean_params, bbox_type='rect', encoder='vit', img_feat_num=768):
        print('Using Vitpose-B!')
        super(HMR_vit, self).__init__()
        self.encoder = ViT(img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,)


        npose = 24 * 6
        nshape = 10
        ncam = 3
        nbbox = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + nbbox + npose + nshape + ncam
        # CUDA Error: an illegal memory access was encountered
        # the above error will occur, if use mobilenet v3 with BN, so don't use BN
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        old_dict = torch.load('ViTPose/vitpose_base_coco_aic_mpii.pth')['state_dict']
        new_dict = OrderedDict([(k.replace('backbone.', ''), v) for k,v in old_dict.items()])
        del new_dict["keypoint_head.final_layer.weight"]
        del new_dict["keypoint_head.final_layer.bias"]

        self.encoder.load_state_dict(new_dict, strict=False)

    def forward(self, x, bbox, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.encoder(x)
        xf = self.avgpool(xf)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, bbox, pred_pose, pred_shape, pred_cam], 1)

            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam
    
class HMR_hrnet_cliff(nn.Module):
    """ SMPL Iterative Regressor with HRNet backbone
    """

    def __init__(self, smpl_mean_params, bbox_type='square', encoder='hr48', img_feat_num=2048):
        print('Using cliff-version HRNet!')
        super(HMR_hrnet_cliff, self).__init__()
        suffix = ''
        if bbox_type=='rect':
            suffix = '_rect'
        config_file = "models/backbones/hrnet/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100{}.yaml".format(suffix)
        update_config(cfg, config_file)
        self.encoder = HighResolutionNet(cfg)
        self.rot_dim = 6

        npose = 24 * self.rot_dim
        nshape = 10
        ncam = 3
        nbbox = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + nbbox + npose + nshape + ncam
        # CUDA Error: an illegal memory access was encountered
        # the above error will occur, if use mobilenet v3 with BN, so don't use BN
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, bbox, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.encoder(x)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, bbox, pred_pose, pred_shape, pred_cam], 1)

            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam
    
class HMR_extraviews(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_extraviews, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox)*(n_extraviews+1) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.regression = [self.fc1, self.drop1, self.fc2, self.drop2, self.decpose, self.decshape, self.deccam]
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        print()

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        print(batch_size, _, h, w)
        xf_cat = []

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        # start = time.time()
        # for view in range(n_extraviews+1):
        #     x = x_all[:, 3*view:3*(view+1)]
        #     x = self.conv1(x)
        #     x = self.bn1(x)
        #     x = self.relu(x)
        #     x = self.maxpool(x)

        #     x1 = self.layer1(x)
        #     x2 = self.layer2(x1)
        #     x3 = self.layer3(x2)
        #     x4 = self.layer4(x3)

        #     xf = self.avgpool(x4)
        #     xf = xf.view(xf.size(0), -1)
        #     xf = torch.cat([xf, bbox_info_all[:, 3*view:3*(view+1)]], 1)
        #     xf_cat.append(xf)

        # xf_cat = torch.cat(xf_cat, 1)
        # print('xf_cat', xf_cat.shape)
        # end = time.time()
        # print("Serial Time:", end - start)

        start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1) #(5*B, 2048)
        xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
        xf_cat = xf_cat.view(batch_size, -1)
        print('xf_cat', xf_cat.shape)
        end = time.time()
        # print("Parallel Time:", end - start)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_cat, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam
    
class HMR_extraviews_hrnet(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, smpl_mean_params, n_extraviews=4, bbox_type='square'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_extraviews_hrnet, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.backbone_out = 480
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear((self.backbone_out + nbbox)*(n_extraviews+1) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.regression = [self.fc1, self.drop1, self.fc2, self.drop2, self.decpose, self.decshape, self.deccam]
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.backbone=hrnet_w32(pretrained_ckpt_path='logs/pretrained_hrnet/pose_hrnet_w32_256x192.pth', downsample=True, use_conv=True)
        # self.backbone=hrnet_w32(pretrained_ckpt_path='', downsample=True, use_conv=True)


        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        print()

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        xf_cat = []

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        xf = self.backbone(x)
        xf = self.avgpool(xf)
        xf = xf.view(xf.size(0), -1)

        xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
        xf_cat = xf_cat.view(batch_size, -1)
        print('xf_cat', xf_cat.shape)
        end = time.time()
        # print("Parallel Time:", end - start)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        
        start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_cat, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam
    
class HMR_fuseviews(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_fuseviews, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.pos_enc = PositionalEncoding()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox*(n_extraviews+1)) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(512 * block.expansion + 2*2*32+2, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_extraviews + 1),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        print()

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = n_extraviews+1

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(batch_size, n_views, -1) #(B, 5, 2048)
        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)
        extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
        main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
        # print(extra_inds, main_inds)
        bbox_trans = bbox_info_all[:, extra_inds, :2] - bbox_info_all[:, main_inds, :2] #(B, 25, 2)
        bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, 2))
        print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)

        xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
        xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)

        # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
        # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
        # print('xf_cat', xf_cat.shape)

        xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
        # print('xf_cat', xf_attention.shape)
        xf_attention = self.attention(xf_attention)
        xf_attention = self.softmax(xf_attention)
        # print('attention', xf_attention.shape)
        xf_out = torch.mul(xf, xf_attention[:, :, None])
        # print(xf_out.shape)
        xf_out = torch.sum(xf_out, dim=1)
        # print('xf_out', xf_out.shape)


        # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
        # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
        # xf_cat = xf_cat.view(batch_size, n_views, -1)
        # print('xf_cat', xf_cat.shape)
        # end = time.time()
        # print("Parallel Time:", end - start)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam

class HMR_sim(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square', wo_enc=False, wo_fuse=False, encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.n_extraviews = n_extraviews
        self.d_in = 3
        self.wo_enc = wo_enc
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in, wo_enc=self.wo_enc)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox*(self.n_extraviews+1)) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(512 * block.expansion + self.pos_enc.d_out, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (self.n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.n_extraviews + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(True),
            nn.Linear(512 * block.expansion, 512 * block.expansion, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = self.n_extraviews+1
        # print('n_views', n_views)

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        # print(xf.shape)
        xf = xf.view(batch_size, n_views, -1) #(B, 5, 2048)

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)
        


        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)

        if not self.wo_fuse:
            extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
            main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
            # print(extra_inds, main_inds)
            bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds, :self.d_in] #(B, 25, 3)
            if not self.wo_enc:
                bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in))
                xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
                xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)
                # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)
            else:
                # print("Not using relative encodings !!")
                xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
                xf_cat = xf # (5*B, 5, 2048)

            

            # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # print('xf_cat', xf_cat.shape)

            xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            xf_out = torch.mul(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = torch.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)


            # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
            # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
            # xf_cat = xf_cat.view(batch_size, n_views, -1)
            # print('xf_cat', xf_cat.shape)
            # end = time.time()
            # print("Parallel Time:", end - start)
        
        else:
            # print("Not using fusion module !!")
            xf_out = xf.view(n_views*batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g

class HMR_sim_catbbx(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square', wo_enc=False, wo_fuse=False, encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim_catbbx, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.n_extraviews = n_extraviews
        self.d_in = 3
        self.wo_enc = wo_enc
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in, wo_enc=self.wo_enc)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + nbbox + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        # self.fuse_fc = nn.Linear(512 * block.expansion + self.pos_enc.d_out, 256)
        self.attention = nn.Sequential(
            nn.Linear(2051, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(True),
            nn.Linear(512 * block.expansion, 512 * block.expansion, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = self.n_extraviews+1
        # print('n_views', n_views)

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        # print(xf.shape)
        xf = xf.view(batch_size, n_views, -1) #(B, 5, 2048)

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)
        


        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)

        if not self.wo_fuse:
            # extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
            # main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
            # # print(extra_inds, main_inds)
            # bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds, :self.d_in] #(B, 25, 3)
            # if not self.wo_enc:
            #     bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in))
            #     xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
            #     xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)
            #     # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)
            # else:
            #     # print("Not using relative encodings !!")
            #     xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
            #     xf_cat = xf # (5*B, 5, 2048)

            

            # # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # # print('xf_cat', xf_cat.shape)

            # xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_catbbx = torch.cat((xf, bbox_info_all), -1) # b, 5, 2051
            xf_attention = self.attention(xf_catbbx).squeeze() #b, 5
            xf_attention = self.softmax(xf_attention)#b,5
            # print('attention', xf_attention.shape)
            xf_out = torch.mul(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = torch.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)


            # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
            # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
            # xf_cat = xf_cat.view(batch_size, n_views, -1)
            # print('xf_cat', xf_cat.shape)
            # end = time.time()
            # print("Parallel Time:", end - start)
        
        
            xf_out = xf.view(n_views*batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_out, bbox_info_all.view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g

class HMR_sim_hrnet(nn.Module):

    def __init__(self, smpl_mean_params, n_extraviews=4, bbox_type='rect', wo_fuse=False, wo_enc=False, encoder='hr32'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim_hrnet, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.d_in = 3
        if encoder == 'hr32':
            self.backbone_out = 480
        elif encoder == 'hr48':
            self.backbone_out = 720
        else:
            raise Exception("Unsupported backbone!")
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.avgpool = nn.AvgPool2d((8,6), stride=1)
        # self.fc1 = nn.Linear((512 * block.expansion + nbbox*(n_extraviews+1)) + npose + 13, 2048)
        if not self.wo_fuse:
            self.fc1 = nn.Linear((self.backbone_out + nbbox*(n_extraviews+1)) + npose + 13, 2048)
        else:
            self.fc1 = nn.Linear((self.backbone_out + nbbox) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(self.backbone_out + self.pos_enc.d_out, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_extraviews + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(self.backbone_out, self.backbone_out),
            nn.BatchNorm1d(self.backbone_out),
            nn.ReLU(True),
            nn.Linear(self.backbone_out, self.backbone_out, bias=False),
            nn.BatchNorm1d(self.backbone_out),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if encoder == 'hr32':
            self.backbone=hrnet_w32(pretrained_ckpt_path='logs/pretrained_hrnet/pose_hrnet_w32_256x192.pth', downsample=True, use_conv=True)
        elif encoder == 'hr48':
            self.backbone=hrnet_w48(pretrained_ckpt_path='logs/pretrained_hrnet/pose_hrnet_w48_256x192.pth', downsample=True, use_conv=True)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = n_extraviews+1

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        xf = self.backbone(x)
        # print('xf:', xf.shape)
        xf = self.avgpool(xf).view(batch_size, n_views, -1)
        # print('xf:', xf.shape)
        

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)
        # print('xf_g:', xf_g.shape)
        


        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)

        if not self.wo_fuse:
            extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
            main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
            # print(extra_inds, main_inds)
            bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds, :self.d_in] #(B, 25, 3)
            bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in))
            # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)

            xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
            xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)

            # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # print('xf_cat', xf_cat.shape)

            xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            xf_out = torch.mul(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = torch.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)
        else:
            print("Not using fusion module !!")
            xf_out = xf.view(n_views*batch_size, -1)


        # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
        # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
        # xf_cat = xf_cat.view(batch_size, n_views, -1)
        # print('xf_cat', xf_cat.shape)
        # end = time.time()
        # print("Parallel Time:", end - start)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            if self.wo_fuse:
                xc = torch.cat([xf_out, bbox_info_all.view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            else:
                xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g
    
class HMR_sim_vit(nn.Module):

    def __init__(self, smpl_mean_params, n_extraviews=4, bbox_type='rect', wo_fuse=False, wo_enc=False, encoder='hr32'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim_vit, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.d_in = 3
        self.backbone = ViT(img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,)
        
        self.backbone_out = 768
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.avgpool = nn.AvgPool2d((8,6), stride=1)
        # self.fc1 = nn.Linear((512 * block.expansion + nbbox*(n_extraviews+1)) + npose + 13, 2048)
        if not self.wo_fuse:
            self.fc1 = nn.Linear((self.backbone_out + nbbox*(n_extraviews+1)) + npose + 13, 2048)
        else:
            self.fc1 = nn.Linear((self.backbone_out + nbbox) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(self.backbone_out + self.pos_enc.d_out, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_extraviews + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(self.backbone_out, self.backbone_out),
            nn.BatchNorm1d(self.backbone_out),
            nn.ReLU(True),
            nn.Linear(self.backbone_out, self.backbone_out, bias=False),
            nn.BatchNorm1d(self.backbone_out),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        old_dict = torch.load('ViTPose/vitpose_base_coco_aic_mpii.pth')['state_dict']
        new_dict = OrderedDict([(k.replace('backbone.', ''), v) for k,v in old_dict.items()])
        del new_dict["keypoint_head.final_layer.weight"]
        del new_dict["keypoint_head.final_layer.bias"]
        self.backbone.load_state_dict(new_dict, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = n_extraviews+1

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        xf = self.backbone(x)
        xf = self.avgpool(xf).view(batch_size, n_views, -1)
        # print('xf:', xf.shape)

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)
        # print('xf_g:', xf_g.shape)
        


        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)

        if not self.wo_fuse:
            extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
            main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
            # print(extra_inds, main_inds)
            bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds, :self.d_in] #(B, 25, 3)
            bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in))
            # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)

            xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
            xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)

            # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # print('xf_cat', xf_cat.shape)

            xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            xf_out = torch.mul(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = torch.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)
        else:
            print("Not using fusion module !!")
            xf_out = xf.view(n_views*batch_size, -1)


        # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
        # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
        # xf_cat = xf_cat.view(batch_size, n_views, -1)
        # print('xf_cat', xf_cat.shape)
        # end = time.time()
        # print("Parallel Time:", end - start)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            if self.wo_fuse:
                xc = torch.cat([xf_out, bbox_info_all.view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            else:
                xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g
    
class HMR_sim_posenc(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square', wo_enc=False, wo_fuse=False, encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim_posenc, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.n_extraviews = n_extraviews
        self.d_in = 3
        self.wo_enc = wo_enc
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in, wo_enc=self.wo_enc, freq_factor=64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox*(self.n_extraviews+1)) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(512 * block.expansion + self.pos_enc.d_out, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (self.n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.n_extraviews + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(True),
            nn.Linear(512 * block.expansion, 512 * block.expansion, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = self.n_extraviews+1
        # print('n_views', n_views)

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        # print(xf.shape)
        xf = xf.view(batch_size, n_views, -1) #(B, 5, 2048)

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)
        # xf = xf_g.view(batch_size, n_views, -1)
        


        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)

        if not self.wo_fuse:
            extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
            main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
            # print(extra_inds, main_inds)
            bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds, :self.d_in] #(B, 25, 3)
            if not self.wo_enc:
                bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in))
                # bbox_trans_emb = bbox_trans.view(-1, self.d_in)
                xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
                xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)
                # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)
            else:
                # print("Not using relative encodings !!")
                xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
                xf_cat = xf # (5*B, 5, 2048)

            

            # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # print('xf_cat', xf_cat.shape)

            xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            xf_out = torch.mul(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = torch.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)


            # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
            # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
            # xf_cat = xf_cat.view(batch_size, n_views, -1)
            # print('xf_cat', xf_cat.shape)
            # end = time.time()
            # print("Parallel Time:", end - start)
        
        else:
            # print("Not using fusion module !!")
            xf_out = xf.view(n_views*batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g
    
class HMR_sim_catenc(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square', wo_enc=False, wo_fuse=False, encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_sim_catenc, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.n_extraviews = n_extraviews
        self.d_in = 3
        self.wo_enc = wo_enc
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in, wo_enc=self.wo_enc)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox*(self.n_extraviews+1)) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.fuse_fc = nn.Linear(512 * block.expansion + 2*self.pos_enc.d_out, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (self.n_extraviews + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.n_extraviews + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(True),
            nn.Linear(512 * block.expansion, 512 * block.expansion, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
        )
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = self.n_extraviews+1
        # print('n_views', n_views)

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        # print(xf.shape)
        xf = xf.view(batch_size, n_views, -1) #(B, 5, 2048)

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)
        


        # print('xf', xf.shape)
        bbox_info_all = bbox_info_all.view(batch_size, n_views, 3) #(B, 5, 3)

        if not self.wo_fuse:
            extra_inds = torch.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1,)
            main_inds = torch.arange(n_views).unsqueeze(0).repeat(n_views,  1).view(-1,)
            # print(extra_inds, main_inds)
            # bbox_trans = bbox_info_all[:, extra_inds, :self.d_in] - bbox_info_all[:, main_inds, :self.d_in] #(B, 25, 3)
            bbox_trans = bbox_info_all[:, :, :self.d_in]
            if not self.wo_enc:
                bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.d_in)).view(batch_size, n_views, -1) #(B, 5, 195)
                # print(bbox_trans_emb.shape)
                extra_embs = bbox_trans_emb[:, extra_inds]
                main_embs = bbox_trans_emb[:, main_inds]
                bbox_trans_emb = torch.cat([extra_embs, main_embs], -1)
                # print(bbox_trans_emb.shape)
                xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
                xf_cat = torch.cat([xf, bbox_trans_emb.view(n_views*batch_size, n_views, -1)], -1) # (5*B, 5, 2048+195)
                # print('bbox_trans_emb',bbox_trans_emb.shape) # (25*B, 195)
            else:
                # print("Not using relative encodings !!")
                xf = xf.repeat(1, n_views, 1).view(n_views*batch_size, n_views, -1) #(5*B, 5, 2048)
                xf_cat = xf # (5*B, 5, 2048)

            

            # xf_cat = torch.cat([xf, bbox_trans[:, :5]], -1) # (B, 5, 2050)
            # xf_cat = torch.cat([xf, bbox_trans_emb.view(batch_size, n_views*n_views, -1)[:, :5]], -1) # (B, 5, 2048+195)
            # print('xf_cat', xf_cat.shape)

            xf_attention = self.fuse_fc(xf_cat).view(n_views*batch_size, -1)
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            xf_out = torch.mul(xf, xf_attention[:, :, None])
            # print(xf_out.shape)
            xf_out = torch.sum(xf_out, dim=1)
            # print('xf_out', xf_out.shape)


            # xf = xf.view(xf.size(0), -1) #(5*B, 2048)
            # xf_cat = torch.cat([xf, bbox_info_all.view(-1, 3)], 1) #(5*B, 2051)
            # xf_cat = xf_cat.view(batch_size, n_views, -1)
            # print('xf_cat', xf_cat.shape)
            # end = time.time()
            # print("Parallel Time:", end - start)
        
        else:
            # print("Not using fusion module !!")
            xf_out = xf.view(n_views*batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g
    
class BertSelfAttention(nn.Module):
    def __init__(self, feature_size, enc_size, num_attention_heads=4, output_attentions=False):
        super(BertSelfAttention, self).__init__()
        hidden_size = feature_size + enc_size
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(self.all_head_size, feature_size)
        self.LayerNorm = BertLayerNorm(feature_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, feature, bbox_enc):
        paddings = torch.zeros_like(bbox_enc, device=bbox_enc.device)
        # print(feature.shape, bbox_enc.shape)
        padded_feature = torch.cat([feature, paddings], -1)
        encoded_feature = torch.cat([feature, bbox_enc], -1)
        mixed_query_layer = self.query(encoded_feature)
        mixed_key_layer = self.key(encoded_feature)
        mixed_value_layer = self.value(padded_feature)
        # print(mixed_key_layer.shape, mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print('attention:', attention_scores.shape)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        out_self = self.out(context_layer)
        outputs = self.LayerNorm(out_self)
        return outputs 

class HMR_msa(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, n_extraviews=4, bbox_type='square', wo_enc=False, wo_fuse=False, encoder=None):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        self.inplanes = 64
        super(HMR_msa, self).__init__()
        npose = 24 * 6
        nbbox = 3
        self.n_extraviews = n_extraviews
        self.d_in = 3
        self.wo_enc = wo_enc
        self.wo_fuse = wo_fuse
        self.pos_enc = PositionalEncoding(d_in=self.d_in, wo_enc=self.wo_enc)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if bbox_type=='square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type=='rect':
            self.avgpool = nn.AvgPool2d((8,6), stride=1)
        self.fc1 = nn.Linear((512 * block.expansion + nbbox*(self.n_extraviews+1)) + npose + 13, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.proj_head = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * block.expansion),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(True),
            nn.Linear(512 * block.expansion, 512 * block.expansion, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.MSA_layer = BertSelfAttention(feature_size=2048, enc_size=195)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x_all, bbox_info_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3, n_extraviews=4):
        # print(x_all.shape)

        batch_size, _, h, w = x_all.shape #(B, 5*c, h, w) #(B, 5*c)
        # print(batch_size, _, h, w)
        n_views = self.n_extraviews+1
        # print('n_views', n_views)

        if init_pose is None:
            init_pose = self.init_pose.expand(n_views*batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(n_views*batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(n_views*batch_size, -1)

        # start = time.time()

        x = x_all.view(-1, 3, h, w) #(5*B, c, h, w)
        # print('x', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        # print(xf.shape)
        xf = xf.view(batch_size, n_views, -1) #(B, 5, 2048)

        xf_ = self.relu(xf.view(n_views*batch_size, -1))
        alpha = self.sigmoid(xf_)
        xf_hidden = torch.mul(xf.view(n_views*batch_size, -1), alpha)
        # print(xf_.shape, alpha.shape, xf_hidden.shape)
        # print(torch.mean(alpha, dim=-1)[0])
        xf_g = self.proj_head(xf_hidden)

        # print('xf', xf.shape)
        bbox_info_flatten = bbox_info_all.view(batch_size*n_views, 3) #(B, 5, 3)
        bbox_trans_enc = self.pos_enc(bbox_info_flatten).view(batch_size, n_views, -1)

        xf_out = self.MSA_layer(xf.view(batch_size, n_views, -1), bbox_trans_enc).view(batch_size*n_views, -1)

        # print("xf_out", xf_out.shape)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        


        # start = time.time()
        for i in range(n_iter):
            xc = torch.cat([xf_out, bbox_info_all.repeat(1, n_views, 1).view(n_views*batch_size, -1), pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views*batch_size, 24, 3, 3)
        end = time.time()
        # print("Regression Time:", end - start)

        return pred_rotmat, pred_shape, pred_cam, xf_g



def hmr(smpl_mean_params, pretrained=True, name='hmr', **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if name == 'hmr': 
        model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_hrnet': 
        model = HMR_hrnet(smpl_mean_params, **kwargs)
    elif name =='hmr_hrnet_cliff':
        print('Using HRNet by cliff')
        model = HMR_hrnet_cliff(smpl_mean_params, **kwargs)
    elif name == 'hmr_vit':
        model = HMR_vit(smpl_mean_params, **kwargs)
    elif name == 'hmr_extraviews': 
        model = HMR_extraviews(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_extraviews_hrnet': 
        model = HMR_extraviews_hrnet(smpl_mean_params, **kwargs)
    elif name == 'hmr_fuseviews': 
        model = HMR_fuseviews(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_sim': 
        model = HMR_sim(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_sim_vit': 
        model = HMR_sim_vit(smpl_mean_params, **kwargs)
    elif name == 'hmr_sim_catbbx': 
        model = HMR_sim_catbbx(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_sim_posenc': 
        model = HMR_sim_posenc(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_sim_catenc': 
        model = HMR_sim_catenc(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_msa': 
        model = HMR_msa(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif name == 'hmr_sim_hrnet': 
        model = HMR_sim_hrnet(smpl_mean_params, **kwargs)
    
    if pretrained and not 'hrnet' in name and not 'vit' in name:
        print('Loading pretrained weights for ResNet backbone...')
        # resnet_imagenet = resnet.resnet50(pretrained=True)
        # model.load_state_dict(resnet_imagenet.state_dict(), strict=False)
        resnet_coco = torch.load('logs/pretrained_resnet/pose_resnet.pth')
        # for k,v in resnet_coco.items():
        #     print(k)
        ## Edit keys in original weights file from CLIFF
        old_dict = resnet_coco['state_dict']
        new_dict = OrderedDict([(k.replace('backbone.', ''), v) for k,v in old_dict.items()])
        model.load_state_dict(new_dict, strict=False)
    # elif pretrained and 'cliff' in name:
        ## Edit keys in original weights file from CLIFF
        
    return model

