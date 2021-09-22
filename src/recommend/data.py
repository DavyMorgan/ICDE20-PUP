#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

import os

import config.const as const_util


class TestDataset(Dataset):

    def __init__(self, rm):

        self.rm = rm
    
    def __getitem__(self, index):

        test_positive_i = self.rm.test_positive[str(index)]
        test_positive_i_np = np.array(test_positive_i, dtype=np.int32)
        positive_i = self.rm.positive[str(index)]
        positive_i_np = np.array(positive_i, dtype=np.int32)
        negative_i_np = self.rm.all_items[np.logical_not(np.isin(self.rm.all_items, positive_i_np))]
        num_positive = len(test_positive_i_np)
        num_negative = len(negative_i_np)
        val_num = num_positive + num_negative
        users = np.array([index for _ in range(val_num)], dtype=np.int32)
        items = np.array([index for _ in range(val_num)], dtype=np.int32)
        items[:num_positive] = test_positive_i_np[:]
        items[num_positive:] = negative_i_np[:]
        cats = self.rm.cats[items]
        prices = self.rm.prices[items]

        users = torch.LongTensor(users)
        items = torch.LongTensor(items)
        cats = torch.LongTensor(cats)
        prices = torch.LongTensor(prices)

        return users, items, cats, prices, num_positive

    def __len__(self):

        return len(self.rm.test_positive)


class FactorizationDataset(Dataset):

    def __init__(self, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        self.user = user
        self.item_p = item_p
        self.item_n = item_n
        self.cat_p = cat_p
        self.cat_n = cat_n
        self.price_p = price_p
        self.price_n = price_n
    
    def __getitem__(self, index):

        return self.user[index], self.item_p[index], self.item_n[index], self.cat_p[index], self.cat_n[index], self.price_p[index], self.price_n[index]
    
    def __len__(self):

        return self.user.size(0)


class FactorizationDataLoaderGenerator(object):

    def __init__(self, datafile_prefix):

        self.datafile_prefix = datafile_prefix
    
    def generate(self, epoch, batch_size, num_workers):

        self.sample_path = os.path.join(self.datafile_prefix, const_util.train_data)
        self.sample_path = os.path.join(self.sample_path, 'epoch_{}'.format(epoch))

        self.load_data()

        self.make_dataset()

        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    def load_data(self):

        self.user = torch.load(os.path.join(self.sample_path, const_util.user))
        self.item_p = torch.load(os.path.join(self.sample_path, const_util.item_p))
        self.item_n = torch.load(os.path.join(self.sample_path, const_util.item_n))
        self.cat_p = torch.load(os.path.join(self.sample_path, const_util.cat_p))
        self.cat_n = torch.load(os.path.join(self.sample_path, const_util.cat_n))
        self.price_p = torch.load(os.path.join(self.sample_path, const_util.price_p))
        self.price_n = torch.load(os.path.join(self.sample_path, const_util.price_n))
    
    def make_dataset(self):

        self.dataset = FactorizationDataset(self.user, self.item_p, self.item_n, self.cat_p, self.cat_n, self.price_p, self.price_n)


class PriceLevelFactorizationDataLoaderGenerator(object):

    def __init__(self, datafile_prefix, price_level):

        self.datafile_prefix = datafile_prefix
        self.price_level = price_level
    
    def generate(self, epoch, batch_size, num_workers):

        self.sample_path = os.path.join(self.datafile_prefix, const_util.train_data)
        self.sample_path = os.path.join(self.sample_path, 'epoch_{}'.format(epoch))

        self.load_data()

        self.make_dataset()

        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    def load_data(self):

        self.user = torch.load(os.path.join(self.sample_path, const_util.user))
        self.item_p = torch.load(os.path.join(self.sample_path, const_util.item_p))
        self.item_n = torch.load(os.path.join(self.sample_path, const_util.item_n))
        self.cat_p = torch.load(os.path.join(self.sample_path, const_util.cat_p))
        self.cat_n = torch.load(os.path.join(self.sample_path, const_util.cat_n))
        self.price_p = torch.load(os.path.join(self.sample_path, const_util.price_p))
        self.price_n = torch.load(os.path.join(self.sample_path, const_util.price_n))
    
    def make_dataset(self):

        self.dataset = FactorizationDataset(self.user, self.item_p, self.item_n, self.cat_p, self.cat_n, self.price_p, self.price_n)


class RankPriceLevelFactorizationDataLoaderGenerator(object):

    def __init__(self, datafile_prefix, price_level):

        self.datafile_prefix = datafile_prefix
        self.price_level = price_level
    
    def generate(self, epoch, batch_size, num_workers):

        self.sample_path = os.path.join(self.datafile_prefix, const_util.train_data)
        self.sample_path = os.path.join(self.sample_path, 'epoch_{}'.format(epoch))

        self.load_data()

        self.make_dataset()

        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    def load_data(self):

        self.user = torch.load(os.path.join(self.sample_path, const_util.user))
        self.item_p = torch.load(os.path.join(self.sample_path, const_util.item_p))
        self.item_n = torch.load(os.path.join(self.sample_path, const_util.item_n))
        self.cat_p = torch.load(os.path.join(self.sample_path, const_util.cat_p))
        self.cat_n = torch.load(os.path.join(self.sample_path, const_util.cat_n))
        self.price_p = torch.load(os.path.join(self.sample_path, str(self.price_level) + '_rank_' + const_util.price_p))
        self.price_n = torch.load(os.path.join(self.sample_path, str(self.price_level) + '_rank_' + const_util.price_n))
    
    def make_dataset(self):

        self.dataset = FactorizationDataset(self.user, self.item_p, self.item_n, self.cat_p, self.cat_n, self.price_p, self.price_n)
