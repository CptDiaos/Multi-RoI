#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import os
import config as cfg
import numpy as np
from datasets.preprocess import pw3d_extract, \
                                mpi_inf_3dhp_extract, \
                                lsp_dataset_extract,\
                                lsp_dataset_original_extract, \
                                hr_lspet_extract, \
                                mpii_extract, \
                                coco_extract, \
                                coco_plus_smpl, \
                                coco_plus_eft, \
                                mpii_plus_smpl, \
                                h36m_extract, \
                                h36m_train_extract, \
                                h36m_plus_smpl_train, \
                                h36m_plus_smpl_test, \
                                pw3d_reproj_extract, \
                                h36m_mosh_extract,\
                                agora_extract,\
                                pack_data,\
                                agora2spin_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = cfg.OPENPOSE_PATH

    if args.train_files:
        # pw3d_reproj_extract(cfg.PW3D_ROOT, out_path, phase='train')
        # pw3d_extract(cfg.PW3D_ROOT, out_path)
        # h36m_train_extract(cfg.H36M_ROOT, openpose_path, out_path, extract_img=True)
        # h36m_plus_smpl_train(dataset_path=os.path.join(out_path, 'h36m_train.npz'), smpl_path=os.path.join(cfg.H36M_ROOT, 'h36m_mosh_train.npz'), out_path=os.path.join(cfg.H36M_ROOT, 'h36m_smpl_train.npz'))
        # MPI-INF-3DHP dataset preprocessing (training set)
        # mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=False, static_fits=cfg.STATIC_FITS_DIR)
        # coco_plus_smpl(dataset_path=os.path.join(out_path, 'coco_2014_train.npz'), smpl_path=os.path.join(cfg.COCO_ROOT, 'coco2014part_cliffGT.npz'), out_path=os.path.join(out_path, 'coco_2014_smpl_train.npz'))
        # mpii_plus_smpl(dataset_path=os.path.join(out_path, 'mpii_train.npz'), smpl_path=os.path.join(cfg.MPII_ROOT, 'mpii_cliffGT.npz'), out_path=os.path.join(out_path, 'mpii_smpl_w3d_train.npz'))
        # LSP dataset original preprocessing (training set)
        # lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, openpose_path, out_path)
        # h36m_mosh_extract(cfg.H36M_ROOT, out_path, phase='train')
        # agora_extract(cfg.AGORA_ROOT, out_path, phase='train')
        agora2spin_extract(cfg.AGORA_ROOT, out_path, phase='train')
        # pack_data(vertex_save_dir=os.path.join(out_path, 'agora_train_cam.npz'), data_folder=cfg.AGORA_ROOT, split='train', annots_path=os.path.join(cfg.AGORA_ROOT, 'annos'))

        # LSP Extended training set preprocessing - HR version
        # hr_lspet_extract(cfg.LSPET_ROOT, openpose_path, out_path)

        # MPII dataset preprocessing
        # mpii_extract(cfg.MPII_ROOT, openpose_path, out_path)

        # COCO dataset prepreocessing
        # coco_extract(cfg.COCO_ROOT, openpose_path, out_path)
        pass

    if args.eval_files:
        # pw3d_reproj_extract(cfg.PW3D_ROOT, out_path, phase='validation')
        # Human3.6M preprocessing (two protocols)
        # h36m_extract(cfg.H36M_ROOT, out_path, protocol=1, extract_img=False)
        # h36m_extract(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)
        # h36m_plus_smpl_test(dataset_path=os.path.join(out_path, 'h36m_train.npz'), smpl_path=os.path.join(cfg.H36M_ROOT, 'h36m_mosh_train.npz'), out_path=os.path.join(cfg.H36M_ROOT, 'h36m_smpl_train.npz'))
        # h36m_plus_smpl_test(dataset_path=os.path.join(out_path, 'h36m_train.npz'), smpl_path=os.path.join(cfg.H36M_ROOT, 'h36m_mosh_train.npz'), out_path=os.path.join(cfg.H36M_ROOT, 'h36m_smpl_train.npz'))
        
        # MPI-INF-3DHP dataset preprocessing (test set)
        # mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'test')
        
        # 3DPW dataset preprocessing (test set)
        # pw3d_extract(cfg.PW3D_ROOT, out_path)
        # coco_extract(cfg.COCO_ROOT, openpose_path, out_path, phase='val')
        # coco_plus_eft(dataset_path=os.path.join(out_path, 'coco_2014_val.npz'), smpl_path=os.path.join(cfg.COCO_ROOT, 'COCO2014-Val-ver10.json'), out_path=os.path.join(out_path, 'coco_2014_smpl_val.npz'))

        # LSP dataset preprocessing (test set)
        # lsp_dataset_extract(cfg.LSP_ROOT, out_path)
        # h36m_mosh_extract(cfg.H36M_ROOT, out_path, phase='val', protocol='p2', category='Directions')
        # for cate in ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting', 'Waiting', 'Walk']:
        #     h36m_mosh_extract(cfg.H36M_ROOT, out_path, phase='val', protocol='p2', category=cate)
        agora2spin_extract(cfg.AGORA_ROOT, out_path, phase='validation')
        # pack_data(vertex_save_dir=os.path.join(cfg.AGORA_ROOT, 'verts'), data_folder=cfg.AGORA_ROOT, split='validation', annots_path=os.path.join(cfg.AGORA_ROOT, 'annos'))
        pass


    # h36m = np.load(os.path.join(out_path, 
    #     'h36m_valid_protocol2_w2d.npz'))
    # h36m = dict(h36m)
    # names = h36m['imgname']
    # actions = []
    # cams = []
    # seqs_len = {}
    # # print(names)
    # # for name in names:
    # #     seq_name = name.split('/')[-1]
    # #     action, camera, _ = seq_name.split('.')
    # #     action = '_'.join(action.split('_')[1:])
    # #     if not action in actions:
    # #         actions.append(action)
    # #     print(actions)
    # import glob
    # num = 0
    # for imgname in glob.glob(os.path.join(cfg.H36M_ROOT,'images/*.jpg')):
    #     seq_name = imgname.split('/')[-1]
    #     action, camera, _ = seq_name.split('.')
    #     sub = action.split('_')[0]
    #     action = '_'.join(action.split('_')[1:])
    #     cam = camera.split('_')[0]
    #     seq_name = '.'.join(('_'.join((sub, action)), cam))
    #     if not seq_name in seqs_len.keys():
    #         seqs_len[seq_name] = 1
    #     else:
    #         seqs_len[seq_name] = seqs_len[seq_name]+1
    #     num = num+1
    #     # print(imgname)
    # # print(seqs_len)
    # sum = 0
    # for k,v in enumerate(seqs_len.items()):
    #     sum = sum + v[1]

    # print(num, sum)

    # pw3d_lcz = np.load(os.path.join(out_path, 
    #     '3dpw_test_w2d_smpl3d_gender.npz'))
    # pw3d_lcz = dict(pw3d_lcz)
    # names_lcz = pw3d_lcz['imgname']
    # # names_lcz = [name.replace('imageFrames/', '') for name in names_lcz]
    # kp2d_lcz = pw3d_lcz['part']
    # pose_lcz = pw3d_lcz['pose']
    # beta_lcz = pw3d_lcz['shape']
    # center_lcz = pw3d_lcz['center']
    # # has_smpl_lcz = pw3d_lcz['has_smpl']

    # pw3d_fmx = np.load(os.path.join(out_path, 
    #     '3dpw_test_bedlam.npz'))
    # pw3d_fmx = dict(pw3d_fmx)
    # print(pw3d_lcz.keys(), pw3d_fmx.keys())
    # names_fmx = pw3d_fmx['imgname']
    # # names_fmx = [for name in names_fmx]
    # kp2d_fmx = pw3d_fmx['part']
    # pose_fmx = pw3d_fmx['pose']
    # # pose_smplx_fmx = pw3d_fmx['smplx_pose']
    # beta_fmx = pw3d_fmx['shape']
    # center_fmx = pw3d_fmx['center']
    # has_smpl_fmx= pw3d_fmx['has_smpl']
    # # cam_fmx = pw3d_fmx['cam_int']
    # # print(cam_fmx.shape, cam_fmx[0])
    # np.set_printoptions(threshold=1000)
    # # print(pose_fmx.shape)
    # names_new = []
    # # for i in range(len(names_fmx)):
    # #     prefx, appdx = names_fmx[i].split('/video')
    # #     name_new = '/imageFrames'.join([prefx, ''.join(['/video', appdx])])
    # #     names_new.append(name_new)
    # # print(np.all(pose_fmx==pose_lcz))
    # # for i in range(len(names_fmx)):
    # #     if names_lcz[i]==names_new[i]:
    # #         idx=np.full((1, 1), i)
    # #     else:
    # #         idx = np.where(names_lcz==names_fmx[i])
    # #     print(idx)
    # #     for valid_idx in idx[0]:
    # #         # print(valid_idx)
    # #         print(beta_lcz[valid_idx], beta_fmx[i])
    # #         if np.all(beta_lcz[valid_idx] == beta_fmx[i]):
    # #             print('lcz', valid_idx)
    # #             print('fmx', i)
    # #             # print(np.where(pose_lcz[valid_idx]!=pose_fmx[i]))
    # #             # print(kp2d_fmx[i].shape)
    # #             print(np.where(kp2d_lcz[valid_idx]!=kp2d_fmx[i]))
    # #             np.set_printoptions(threshold=1000)
    # #             print(pose_lcz[valid_idx,:]-pose_fmx[i,:])
    # print(np.all(names_lcz==names_fmx))
    # print(np.all(names_lcz==names_fmx))
    # print(np.all(pose_fmx==pose_lcz))
    # print(np.where(pose_fmx!=pose_lcz))
    # print(np.all(center_fmx==center_lcz))
    # print(np.all(has_smpl_fmx==has_smpl_lcz))



