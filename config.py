"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = '../data/human3.6m'
# LSP_ROOT = '../data/lsp_dataset'
# LSP_ORIGINAL_ROOT = '../data/lsp_dataset_original'
# LSPET_ROOT = '../data/hr-lspet'
MPII_ROOT = '../data/mpii_human_pose_v1'
COCO_ROOT = '../data/coco/coco2014'
MPI_INF_3DHP_ROOT = '../data/mpii_3d'
PW3D_ROOT = '../data/3DPW'
# UPI_S1H_ROOT = ''
# AGORA_ROOT = '../data/AGORA'
# CUSTOM_ROOT = './demo/demo_data/images'

# H36M_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/human3.6m'
LSP_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/lsp_dataset'
LSP_ORIGINAL_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/lsp_dataset_original'
LSPET_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/hr-lspet'
# MPII_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/mpii_human_pose_v1'
# COCO_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/coco/coco2014'
# MPI_INF_3DHP_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/mpii_3d'
# PW3D_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/3DPW'
UPI_S1H_ROOT = ''
AGORA_ROOT = '/media/lab345/Elements SE/lcz/data/img_database/AGORA'
CUSTOM_ROOT = './demo/demo_data/images'



# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_mosh_val_p1.npz'),
                  #  'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_w2d.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_mosh_val_p2.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_w2d_smpl3d_gender.npz'),
                   '3dpw_val': join(DATASET_NPZ_PATH, '3dpw_validation_w2d_smpl3d_gender.npz'),
                   'agora': join(DATASET_NPZ_PATH, 'agora2spin_validation_cam_aligned_w_cams_smil_kid.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_smpl_val.npz'),
                   'custom': './demo/demo_data/annotations'
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_smpl_w3d_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_smpl_train.npz'),
                  #  'coco': join(DATASET_NPZ_PATH, 'coco_2014_smpl_train_2-4-8.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                  #  'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train_wo_smpl.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train_name_revised.npz'),
                  #  '3dpw': join(DATASET_NPZ_PATH, '3dpw_train_bedlam.npz')
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_train_w2d_smpl3d_gender.npz'),
                   'agora': join(DATASET_NPZ_PATH, 'agora_train_cam_aligned_w_cams_smil_kid_only.npz'),
                  # 'agora': join(DATASET_NPZ_PATH, 'agora2spin_train_cam_aligned_w_cams_filtered_60_smil_kid_multibbox.npz'),

                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   'agora': AGORA_ROOT,
                   'custom':CUSTOM_ROOT
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
SMPL_KID_TEMP_DIR = 'data/smpl/smpl_kid_template.npy'
SMPL_SEG_PATH = '../data/smpl_vert_6segmentation.json'
