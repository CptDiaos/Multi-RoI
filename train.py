from utils import TrainOptions
from train import Trainer
import os
import numpy as np
import torch
import random

if __name__ == '__main__':
    # seed = 2023
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()