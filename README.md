# Multi-RoI
Official Implementation of **Multi-RoI Human Mesh Recovery with Camera Consistency and Contrastive Losses**
Yongwei Nie, Changzhen Liu, Chengjiang Long, Qing Zhang, Guiqing Li, Hongmin Cai*

**[[Paper]()][[Project Page]()]**

## News :triangular_flag_on_post:

## Instructions
1. To run our code, you need to download raw images for each dataset(Human 3.6M, 3DPW, MPI-INF-3DHP, MPII and COCO 2014) and necessary SMPL files from official websites respectively.
2. Download pretrained weights for ResNet-50 and HRNet-w48.
3. Put these data following the directory structure as below.
```
${ROOT}
|-- data
    smpl_mean_params.npz
    |-- ckpt
        |-- hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
        |-- res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt
        |-- hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
|-- mmdetection
    |-- checkpoints
        |-- yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
|-- mmtracking
    |-- checkpoints
        |-- bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
```
4. Use provided scripts preprocess_datasets.py to extract images and annotations.
```
   python preprocess_datasets.py --train_files\eval_files 
```
## Codes
### Train
1. Pull our code.
2. Download necessary files and organize them according to Instructions.
3. Use this command as example:
   ```
   python train.py --pretrained_checkpoint logs/train_sim_full_wo3dpw_h36mp1_shift_w_rescale_hr48/checkpoints/previous_16_6000_42.4_30.5.pt --encoder hr48 --name train_sim_full_ft_rerun --rescale_bbx --shift_center --train_dataset 3dpw --eval_dataset 3dpw --bbox_type rect --batch_size 20 --use_extraviews --n_views 5 --lr 1e-5
   ```
### Evaluation
Coming soon!
