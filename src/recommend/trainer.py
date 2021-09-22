#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import logging

from tqdm import tqdm

import recommender

import numpy as np
import torch
import torch.optim as optim

class Trainer(object):

    def __init__(self, flags_obj, cm, vm):

        self.cm = cm
        self.vm = vm
        self.name = flags_obj.name
        self.model = flags_obj.model
        self.dataset = flags_obj.dataset
        self.epochs = flags_obj.epochs
        self.lr = flags_obj.lr
        self.weight_decay = flags_obj.weight_decay
        self.dropout = flags_obj.dropout
        self.batch_size = flags_obj.batch_size
        self.lr_decay_epochs = flags_obj.lr_decay_epochs
        self.num_workers = flags_obj.num_workers
        self.datafile_prefix = flags_obj.datafile_prefix
        self.output = flags_obj.output
        self.set_recommender(flags_obj, cm.workspace)
        self.recommender.transfer_model()
    
    def set_recommender(self, flags_obj, workspace):

        self.recommender = self.cm.set_recommender(flags_obj, workspace, self.cm)
    
    def train(self):

        self.set_dataloader_generator()
        self.set_optimizer()
        self.set_scheduler()

        for epoch in range(self.epochs):

            self.train_one_epoch(epoch)
            self.recommender.save_ckpt(epoch)
            self.scheduler.step()
    
    def set_dataloader_generator(self):

        self.generator = self.recommender.get_dataloader_generator()
    
    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()
    
    def set_scheduler(self):

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_epochs, gamma=0.1)
    
    def train_one_epoch(self, epoch):

        data_loader = self.generator.generate(epoch, self.batch_size, self.num_workers)

        running_loss = 0.0
        total_loss = 0.0
        num_batch = len(data_loader)
        distances = np.zeros(num_batch)

        logging.info('learning rate : {}'.format(self.optimizer.param_groups[0]['lr']))

        for batch_count, sample in enumerate(tqdm(data_loader)):

            self.optimizer.zero_grad()

            p_score, n_score = self.recommender.inference(sample)

            distances[batch_count] = (p_score - n_score).mean().item()

            loss = self.bpr_loss(p_score, n_score)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:

                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                running_loss = 0.0
        
        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss))
        self.vm.update_line('loss', epoch, total_loss)
        self.vm.update_line('distance', epoch, distances.mean())
    
    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))