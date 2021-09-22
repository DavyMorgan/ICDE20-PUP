# encoding: utf-8
import torch
import multiprocessing as mp
from multiprocessing import Process
import copy
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import os


def get_bridge(bridge_prefix):

    with open(bridge_prefix + 'item_index.json', 'r') as f:
        item_index = json.loads(f.read())
    with open(bridge_prefix + 'item_index_invert.json', 'r') as f:
        item_index_invert = json.loads(f.read())
    with open(bridge_prefix + 'item_cat.json', 'r') as f:
        item_cat = json.loads(f.read())
    with open(bridge_prefix + 'cat_index.json', 'r') as f:
        cat_index = json.loads(f.read())
    with open(bridge_prefix + 'item_lux.json', 'r') as f:
        item_lux = json.loads(f.read())
    with open(bridge_prefix + 'user_index.json', 'r') as f:
        user_index = json.loads(f.read())
    with open(bridge_prefix + 'user_index_invert.json', 'r') as f:
        user_index_invert = json.loads(f.read())

    return item_index, item_index_invert, item_cat, cat_index, item_lux, user_index, user_index_invert


def split_train_val(user_index_invert, item_index, prefix, bridge_prefix, from_file):

    if not from_file:

        positive = {}
        train_positive = {}
        val_positive = {}
        test_positive = {}

        val_ratio = 0.2
        test_ratio = 0.2

        action_log = pd.read_csv(bridge_prefix + 'action_log.csv')
        action_log = action_log.drop(labels=['Unnamed: 0'], axis=1)
        action_log = action_log.rename(columns={'business_id': 'item_id'})

        print('splitting dataset!')
        for index, user_id in tqdm(user_index_invert.items()):

            interaction = action_log[action_log['user_id'] == user_id]
            num_positive = interaction['item_id'].count()
            interaction = interaction.sort_values(by=['date'])
            interaction.reset_index(drop=True, inplace=True)

            positive_i = interaction['item_id'].tolist()

            num_val = int(round(val_ratio * num_positive))
            num_test = int(round(test_ratio * num_positive))
            num_train = num_positive - num_val - num_test

            train_positive_i = positive_i[:num_train]
            val_positive_i = positive_i[num_train:num_train + num_val]
            test_positive_i = positive_i[num_train + num_val:]

            positive[index] = [item_index[str(item_id)] for item_id in positive_i]
            train_positive[index] = [item_index[str(item_id)] for item_id in train_positive_i]
            val_positive[index] = [item_index[str(item_id)] for item_id in val_positive_i]
            test_positive[index] = [item_index[str(item_id)] for item_id in test_positive_i]

        with open(prefix + 'positive.json', 'w') as f:
            f.write(json.dumps(positive))
        with open(prefix + 'train_positive.json', 'w') as f:
            f.write(json.dumps(train_positive))
        with open(prefix + 'val_positive.json', 'w') as f:
            f.write(json.dumps(val_positive))
        with open(prefix + 'test_positive.json', 'w') as f:
            f.write(json.dumps(test_positive))

    else:

        with open(prefix + 'positive.json', 'r') as f:
            positive = json.loads(f.read())
        with open(prefix + 'train_positive.json', 'r') as f:
            train_positive = json.loads(f.read())
        with open(prefix + 'val_positive.json', 'r') as f:
            val_positive = json.loads(f.read())
        with open(prefix + 'test_positive.json', 'r') as f:
            test_positive = json.loads(f.read())

    return positive, train_positive, val_positive, test_positive


class ResourceManager(object):
    def __init__(self, action_log, user_index_invert, item_index, val_ratio, test_ratio):
        self.action_log = action_log
        self.user_index_invert = user_index_invert
        self.item_index = item_index
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio


class Splitter(Process):

    def __init__(self, name, users, rm, d, lock):
        super(Splitter, self).__init__()
        self.name = str(name)
        self.users = users
        self.num_users = len(users)
        self.rm = rm
        self.d = d
        self.lock = lock
        self.positive = {}
        self.train_positive = {}
        self.val_positive = {}
        self.test_positive = {}

    def run(self):

        print('Splitter {} start working!'.format(self.name))

        for count, user_id in enumerate(self.users):
            if count % (self.num_users // 10) == (self.num_users // 10) - 1:
                print('Splitter {} is working at {} / {}'.format(self.name, count, self.num_users))
            interaction = self.rm.action_log[self.rm.action_log['user_id'] == self.rm.user_index_invert[str(user_id)]]
            num_positive = interaction['item_id'].count()
            interaction = interaction.sort_values(by=['date'])
            interaction.reset_index(drop=True, inplace=True)

            positive_i = interaction['item_id'].tolist()

            num_val = int(round(self.rm.val_ratio * num_positive))
            num_test = int(round(self.rm.test_ratio * num_positive))
            num_train = num_positive - num_val - num_test

            train_positive_i = positive_i[:num_train]
            val_positive_i = positive_i[num_train:num_train + num_val]
            test_positive_i = positive_i[num_train + num_val:]

            self.positive[str(user_id)] = [self.rm.item_index[str(item_id)] for item_id in positive_i]
            self.train_positive[str(user_id)] = [self.rm.item_index[str(item_id)] for item_id in train_positive_i]
            self.val_positive[str(user_id)] = [self.rm.item_index[str(item_id)] for item_id in val_positive_i]
            self.test_positive[str(user_id)] = [self.rm.item_index[str(item_id)] for item_id in test_positive_i]

        with self.lock:
            fake_positive = self.d['positive']
            fake_positive.update(self.positive)
            self.d['positive'] = fake_positive

            fake_train_positive = self.d['train_positive']
            fake_train_positive.update(self.train_positive)
            self.d['train_positive'] = fake_train_positive

            fake_val_positive = self.d['val_positive']
            fake_val_positive.update(self.val_positive)
            self.d['val_positive'] = fake_val_positive

            fake_test_positive = self.d['test_positive']
            fake_test_positive.update(self.test_positive)
            self.d['test_positive'] = fake_test_positive


def multi_split_train_val(num_workers, user_index_invert, item_index, prefix, bridge_prefix, from_file):

    if not from_file:

        val_ratio = 0.2
        test_ratio = 0.2

        action_log = pd.read_csv(bridge_prefix + 'action_log.csv')
        action_log = action_log.drop(labels=['Unnamed: 0'], axis=1)
        action_log = action_log.rename(columns={'business_id': 'item_id'})

        rm = ResourceManager(action_log, user_index_invert, item_index, val_ratio, test_ratio)

        lock = mp.Lock()
        manager = mp.Manager()
        d = manager.dict()
        d['positive'] = {}
        d['train_positive'] = {}
        d['val_positive'] = {}
        d['test_positive'] = {}

        processes = []

        for i in range(num_workers):
            duty_users = [j for j in range(len(user_index_invert)) if j % num_workers == i]
            p = Splitter(i, duty_users, copy.deepcopy(rm), d, lock)
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        positive = d['positive']
        train_positive = d['train_positive']
        val_positive = d['val_positive']
        test_positive = d['test_positive']

        with open(prefix + 'positive.json', 'w') as f:
            f.write(json.dumps(positive))
        with open(prefix + 'train_positive.json', 'w') as f:
            f.write(json.dumps(train_positive))
        with open(prefix + 'val_positive.json', 'w') as f:
            f.write(json.dumps(val_positive))
        with open(prefix + 'test_positive.json', 'w') as f:
            f.write(json.dumps(test_positive))

    else:

        with open(prefix + 'positive.json', 'r') as f:
            positive = json.loads(f.read())
        with open(prefix + 'train_positive.json', 'r') as f:
            train_positive = json.loads(f.read())
        with open(prefix + 'val_positive.json', 'r') as f:
            val_positive = json.loads(f.read())
        with open(prefix + 'test_positive.json', 'r') as f:
            test_positive = json.loads(f.read())

    return positive, train_positive, val_positive, test_positive


def generate_cat_price(item_index_invert, item_cat, cat_index, item_lux, from_file, prefix):

    if not from_file:
        num_items = len(item_index_invert)

        cats = np.zeros([num_items], dtype=np.int32)
        prices = np.zeros([num_items], dtype=np.int32)

        for i in range(num_items):
            item_id = item_index_invert[str(i)]
            cats[i] = cat_index[item_cat[item_id]]
            prices[i] = item_lux[item_id] - 1

        np.save(prefix + 'cats.npy', cats)
        np.save(prefix + 'prices.npy', prices)

    else:
        cats = np.load(prefix + 'cats.npy')
        prices = np.load(prefix + 'prices.npy')

    return cats, prices


def compute_sales(prefix, positive, num_items, from_file):

    if not from_file:

        temp = []

        print('computing sales!')
        for uid, iids in tqdm(positive.items()):

            temp = temp + iids

        temp = torch.Tensor(temp)

        sales = torch.histc(temp, bins=num_items, min=0, max=num_items - 1)

        torch.save(sales, prefix + 'sales.pth')

    else:

        sales = torch.load(prefix + 'sales.pth')

    return sales
