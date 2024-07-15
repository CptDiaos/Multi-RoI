from __future__ import division
import sys
import time
import os

import torch
from tqdm import tqdm

tqdm.monitor_interval = 0
from torch.utils.tensorboard import SummaryWriter

from utils import CheckpointDataLoader, CheckpointSaver


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """

    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)
        self.best_mpjpe = None
        self.best_pampjpe = None

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint(self.options.checkpoint):
            print('Resuming')
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict,
                                                         checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        print("LOAD")
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model], strict=True)
                    print('Checkpoint loaded')

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs,
                          initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)
            
            # if self.options.acc_loss:
            acc_2dloss = 0
            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    # out = self.train_step(batch)
                    out, loss = self.train_step(batch, epoch, step)
                    acc_2dloss = acc_2dloss + loss['loss_keypoints']
                    self.step_count += 1
                    if self.step_count % 100 ==0:
                        with open(os.path.join(self.options.log_dir, 'acc_loss.txt'), mode='a', encoding='utf-8') as logger:
                            print('Epoch {}-step {}: '.format(epoch, step+1), (acc_2dloss/100), file=logger)
                            acc_2dloss = 0
                    # Tensorboard logging every summary_steps steps
                    if self.options.sum and self.step_count % self.options.summary_steps == 0:
                        # batch_sum = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                        #              next(train_noaug_data_iter).items()}
                        # out_sum = self.train_step(batch_sum)
                        # self.train_summaries(batch_sum, *out_sum)
                        self.train_summaries(batch, out, loss)
                        pass
                    # Save checkpoint every checkpoint_steps steps
                        pass
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                                   self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                                   self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if (step+1) % self.options.test_steps == 0:
                        eval_stats = self.test(epoch)
                        with open(os.path.join(self.options.log_dir, 'test_log.txt'), mode='a', encoding='utf-8') as logger:
                            print('Epoch {}-step {}: '.format(epoch, step+1), eval_stats, file=logger)
                        if (epoch + 1) % self.options.save_epochs == 0:
                            self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                                self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count, is_update=False, eval_stats=eval_stats)
                else:
                    tqdm.write('Timeout reached')
                    self.finalize()
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                                               self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                               self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

                if self.options.viz_debug:
                    time.sleep(15)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint = None
            # save checkpoint after each epoch
            # if (epoch + 1) % 1 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
            if ((epoch + 1) % self.options.save_epochs == 0 or (epoch + 1) % self.options.test_epochs == 0)and (epoch + 1) >= self.options.test_start_epoch:
                eval_stats = self.test(epoch + 1)
                with open(os.path.join(self.options.log_dir, 'test_log.txt'), mode='a', encoding='utf-8') as logger:
                    print('Epoch {}: '.format(epoch+1), eval_stats, file=logger)
                if (epoch + 1) % self.options.save_epochs == 0:
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                            self.options.batch_size, None, self.step_count, is_update=False, eval_stats=eval_stats)
                    
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        raise NotImplementedError('You need to provide a _test method')
