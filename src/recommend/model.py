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


class BasePUP(nn.Module):

    def __init__(self, dropout, alpha, split_dim, gc):

        super(BasePUP, self).__init__()

        self.dropout = dropout
        self.alpha = alpha
        self.split_dim = split_dim
        self.gc = gc
    
    def forward(self, feature, adj, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        x = self.encode(feature, adj)
        pred_p, pred_n = self.decode(x, user, item_p, item_n, cat_p, cat_n, price_p, price_n)

        return pred_p, pred_n
    
    def encode(self, feature, adj, training=True):

        x = self.gc(feature, adj)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=training)

        return x
    
    def test_encode(self, feature, adj):

        return self.encode(feature, adj, False)
    
    def decode(self, x, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        pred_p = self.decode_core(x, user, item_p, cat_p, price_p)
        pred_n = self.decode_core(x, user, item_n, cat_n, price_n)

        return pred_p, pred_n
    
    def decode_core(self, x, user, item, cat, price):

        user_embedding = x[user]
        item_embedding = x[item]
        cat_embedding = x[cat]
        price_embedding = x[price]

        (user_global, user_category) = torch.split(user_embedding, self.split_dim, 1)
        (item_global, _) = torch.split(item_embedding, self.split_dim, 1)
        (_, cat_category) = torch.split(cat_embedding, self.split_dim, 1)
        (price_global, price_category) = torch.split(price_embedding, self.split_dim, 1)

        pred_global = self.fm([user_global, item_global, price_global])
        pred_category = self.fm([user_category, cat_category, price_category])
        scores = pred_global + self.alpha * pred_category

        return scores
    
    def test_decode(self, x, user, item, cat, price):

        return self.decode_core(x, user, item, cat, price)
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class PUP(nn.Module):

    def __init__(self, feature_size, embedding_size, dropout, alpha, split_dim):

        super(PUP, self).__init__()
        gc = GraphConvolution(feature_size, embedding_size)
        self.base_pup = BasePUP(dropout, alpha, split_dim, gc)

    def forward(self, feature, adj, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        pred_p, pred_n= self.base_pup(feature, adj, user, item_p, item_n, cat_p, cat_n, price_p, price_n)

        return pred_p, pred_n
    
    def test_encode(self, feature, adj):

        return self.base_pup.test_encode(feature, adj)
    
    def test_decode(self, x, user, item, cat, price):

        return self.base_pup.test_decode(x, user, item, cat, price)


class PUPMinusC(nn.Module):

    def __init__(self, feature_size, embedding_size, dropout, alpha, split_dim):

        super(PUPMinusC, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.split_dim = split_dim
        self.gc = GraphConvolution(feature_size, embedding_size)
    
    def forward(self, feature, adj, user, item_p, item_n, price_p, price_n):

        x = self.encode(feature, adj)
        pred_p, pred_n = self.decode(x, user, item_p, item_n, price_p, price_n)

        return pred_p, pred_n
    
    def encode(self, feature, adj, training=True):

        x = self.gc(feature, adj)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=training)

        return x
    
    def test_encode(self, feature, adj):

        return self.encode(feature, adj, False)
    
    def decode(self, x, user, item_p, item_n, price_p, price_n):

        pred_p = self.decode_core(x, user, item_p, price_p)
        pred_n = self.decode_core(x, user, item_n, price_n)

        return pred_p, pred_n
    
    def decode_core(self, x, user, item, price):

        user_embedding = x[user]
        item_embedding = x[item]
        price_embedding = x[price]

        scores = self.fm([user_embedding, item_embedding, price_embedding])

        return scores
    
    def test_decode(self, x, user, item, price):

        return self.decode_core(x, user, item, price)
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm


class PUPMinusP(nn.Module):

    def __init__(self, feature_size, embedding_size, dropout, alpha, split_dim):

        super(PUPMinusP, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.split_dim = split_dim
        self.gc = GraphConvolution(feature_size, embedding_size)
    
    def forward(self, feature, adj, user, item_p, item_n, cat_p, cat_n):

        x = self.encode(feature, adj)
        pred_p, pred_n = self.decode(x, user, item_p, item_n, cat_p, cat_n)

        return pred_p, pred_n
    
    def encode(self, feature, adj, training=True):

        x = self.gc(feature, adj)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=training)

        return x
    
    def test_encode(self, feature, adj):

        return self.encode(feature, adj, False)
    
    def decode(self, x, user, item_p, item_n, cat_p, cat_n):

        pred_p = self.decode_core(x, user, item_p, cat_p)
        pred_n = self.decode_core(x, user, item_n, cat_n)

        return pred_p, pred_n
    
    def decode_core(self, x, user, item, cat):

        user_embedding = x[user]
        item_embedding = x[item]
        cat_embedding = x[cat]

        scores = self.fm([user_embedding, item_embedding, cat_embedding])

        return scores
    
    def test_decode(self, x, user, item, cat):

        return self.decode_core(x, user, item, cat)
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm
    

class PUPMinusCP(nn.Module):

    def __init__(self, feature_size, embedding_size, dropout, alpha, split_dim):

        super(PUPMinusCP, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.split_dim = split_dim
        self.gc = GraphConvolution(feature_size, embedding_size)
    
    def forward(self, feature, adj, user, item_p, item_n):

        x = self.encode(feature, adj)
        pred_p, pred_n = self.decode(x, user, item_p, item_n)

        return pred_p, pred_n
    
    def encode(self, feature, adj, training=True):

        x = self.gc(feature, adj)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=training)

        return x
    
    def test_encode(self, feature, adj):

        return self.encode(feature, adj, False)
    
    def decode(self, x, user, item_p, item_n):

        pred_p = self.decode_core(x, user, item_p)
        pred_n = self.decode_core(x, user, item_n)

        return pred_p, pred_n
    
    def decode_core(self, x, user, item):

        user_embedding = x[user]
        item_embedding = x[item]

        scores = self.fm([user_embedding, item_embedding])

        return scores
    
    def test_decode(self, x, user, item):

        return self.decode_core(x, user, item)
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm
