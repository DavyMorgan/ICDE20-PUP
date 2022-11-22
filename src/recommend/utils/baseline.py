#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.init_params()
    
    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)
    
    def forward(self, user, item_p, item_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 1)
        n_score = torch.sum(user * item_n, 1)

        return p_score, n_score
    
    def test_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]
        score = torch.sum(user * item, 1)

        return score


class FM(nn.Module):

    def __init__(self, num_users, num_items, num_cats, num_prices, embedding_size):

        super(FM, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.cats = Parameter(torch.FloatTensor(num_cats, embedding_size))
        self.prices = Parameter(torch.FloatTensor(num_prices, embedding_size))

        self.init_params()
    
    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)
        self.cats.data.uniform_(-stdv, stdv)
        self.prices.data.uniform_(-stdv, stdv)
    
    def forward(self, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]
        cat_p = self.cats[cat_p]
        cat_n = self.cats[cat_n]
        price_p = self.prices[price_p]
        price_n = self.prices[price_n]

        p_score = self.fm([user, item_p, cat_p, price_p])
        n_score = self.fm([user, item_n, cat_n, price_n])

        return p_score, n_score
    
    def test_forward(self, user, item, cat, price):

        user = self.users[user]
        item = self.items[item]
        cat = self.items[cat]
        price = self.prices[price]
        score = self.fm([user, item, cat, price])

        return score
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm


class DeepFM(nn.Module):

    def __init__(self, num_users, num_items, num_cats, num_prices, embedding_size):

        super(DeepFM, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.cats = Parameter(torch.FloatTensor(num_cats, embedding_size))
        self.prices = Parameter(torch.FloatTensor(num_prices, embedding_size))

        self.fc1 = nn.Linear(4 * embedding_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        self.init_params()
    
    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)
        self.cats.data.uniform_(-stdv, stdv)
        self.prices.data.uniform_(-stdv, stdv)
    
    def forward(self, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]
        cat_p = self.cats[cat_p]
        cat_n = self.cats[cat_n]
        price_p = self.prices[price_p]
        price_n = self.prices[price_n]

        p_score_fm = self.fm([user, item_p, cat_p, price_p])
        n_score_fm = self.fm([user, item_n, cat_n, price_n])

        p_dnn = torch.cat((user, item_p, cat_p, price_p), 1)
        n_dnn = torch.cat((user, item_n, cat_n, price_n), 1)

        p_dnn = self.fc1(p_dnn)
        p_dnn = F.dropout(p_dnn)
        p_dnn = F.relu(p_dnn)
        p_dnn = self.fc2(p_dnn)
        p_dnn = F.dropout(p_dnn)
        p_dnn = F.relu(p_dnn)
        p_score_dnn = torch.squeeze(self.fc3(p_dnn))

        n_dnn = self.fc1(n_dnn)
        n_dnn = F.dropout(n_dnn)
        n_dnn = F.relu(n_dnn)
        n_dnn = self.fc2(n_dnn)
        n_dnn = F.dropout(n_dnn)
        n_dnn = F.relu(n_dnn)
        n_score_dnn = torch.squeeze(self.fc3(n_dnn))

        p_score = p_score_fm + p_score_dnn
        n_score = n_score_fm + n_score_dnn

        return p_score, n_score
    
    def test_forward(self, user, item, cat, price):

        user = self.users[user]
        item = self.items[item]
        cat = self.items[cat]
        price = self.prices[price]
        score_fm = self.fm([user, item, cat, price])

        dnn = torch.cat((user, item, cat, price), 1)
        dnn = self.fc1(dnn)
        dnn = F.relu(dnn)
        dnn = self.fc2(dnn)
        dnn = F.relu(dnn)
        score_dnn = torch.squeeze(self.fc3(dnn))

        score = score_fm + score_dnn

        return score
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm
