# Multi-RoI
Official Implementation of Multi-RoI Human Mesh Recovery with Camera Consistency and Contrastive Losses



**[[Paper]()][[Project Page]()]**

## News :triangular_flag_on_post:

## Instructions
Coming soon!

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
