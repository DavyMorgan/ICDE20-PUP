#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import data
import model
import utils
import config.const as const_util

import os


class Recommender(object):

    def __init__(self, flags_obj, workspace, cm):

        self.cm = cm
        self.model_name = flags_obj.model
        self.num_users = flags_obj.num_users
        self.num_items = flags_obj.num_items
        self.embedding_size = flags_obj.embedding_size
        self.datafile_prefix = flags_obj.datafile_prefix
        self.lr = flags_obj.lr
        self.set_device(flags_obj)
        self.set_model(flags_obj)
        self.workspace = workspace
    
    def set_device(self, flags_obj):

        self.device  = self.cm.set_device(flags_obj)
    
    def set_model(self, flags_obj):

        pass
    
    def transfer_model(self):

        self.model = self.model.to(self.device)
    
    def save_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), model_path)
    
    def load_ckpt(self):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = os.path.join(ckpt_path, const_util.model)
        self.model.load_state_dict(torch.load(model_path))
    
    def get_dataloader_generator(self, datafile_prefix):

        pass
    
    def get_optimizer(self, lr, weight_decay):

        pass
    
    def inference(self, sample):

        pass
    
    def test_inference(self, sample):

        pass
    
    def prepare_test(self):

        pass


class MFRecommender(Recommender):

    def __init__(self, flags_obj, workspace, cm):

        super(MFRecommender, self).__init__(flags_obj, workspace, cm)
        self.weight_decay = flags_obj.weight_decay
    
    def set_model(self, flags_obj):

        self.model = model.MF(self.num_users, self.num_items, self.embedding_size)
    
    def get_dataloader_generator(self):

        return data.FactorizationDataLoaderGenerator(self.datafile_prefix)
    
    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.99))
    
    def inference(self, sample):
        
        user, item_p, item_n, _, _, _, _ = sample
        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        p_score, n_score = self.model(user, item_p, item_n)

        return p_score, n_score
    
    def test_inference(self, sample):

        users, items, _, _, num_positive = sample
        users = torch.squeeze(users.to(self.device))
        items = torch.squeeze(items.to(self.device))
        scores = self.model.test_forward(users, items)

        return scores, num_positive


class BasePUPRecommender(Recommender):

    def __init__(self, flags_obj, workspace, cm):

        super(BasePUPRecommender, self).__init__(flags_obj, workspace, cm)
        self.weight_decay = flags_obj.weight_decay
        self.num_cats = flags_obj.num_cats
        self.num_prices = flags_obj.num_prices
        self.set_gm(flags_obj)
        self.generate_transfer_feature_adj(flags_obj)
    
    def set_gm(self, flags_obj):

        pass
    
    def generate_transfer_feature_adj(self, flags_obj):

        self.gm.generate_id_feature()
        self.gm.generate_adj(flags_obj)
        self.gm.transfer_data(self.device)
    
    def set_model(self, flags_obj):

        self.set_pup_hyper_params(flags_obj)
        self.set_pup_model()

    def set_pup_model(self):

        pass
    
    def set_pup_hyper_params(self, flags_obj):

        self.set_feature_size(flags_obj)
        self.dropout = flags_obj.dropout
        self.alpha = flags_obj.alpha
        self.split_dim = flags_obj.split_dim
    
    def set_feature_size(self, flags_obj):

        self.feature_size = flags_obj.num_users + flags_obj.num_items + flags_obj.num_cats + flags_obj.num_prices
    
    def get_dataloader_generator(self):

        return data.FactorizationDataLoaderGenerator(self.datafile_prefix)
    
    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.99))
    
    def inference(self, sample):

        user, item_p, item_n, cat_p, cat_n, price_p, price_n = sample
        user = user.to(self.device)
        item_p = item_p.to(self.device) + self.num_users
        item_n = item_n.to(self.device) + self.num_users
        cat_p = cat_p.to(self.device) + self.num_users + self.num_items
        cat_n = cat_n.to(self.device) + self.num_users + self.num_items
        price_p = price_p.to(self.device) + self.num_users + self.num_items + self.num_cats
        price_n = price_n.to(self.device) + self.num_users + self.num_items + self.num_cats
        p_score, n_score = self.model(self.gm.feature, self.gm.adj, user, item_p, item_n, cat_p, cat_n, price_p, price_n)

        return p_score, n_score
    
    def prepare_test(self):

        self.output = self.model.test_encode(self.gm.feature, self.gm.adj)
    
    def test_inference(self, sample):

        users, items, cats, prices, num_positive = sample
        users = torch.squeeze(users.to(self.device))
        items = torch.squeeze(items.to(self.device)) + self.num_users
        cats = torch.squeeze(cats.to(self.device)) + self.num_users + self.num_items
        prices = torch.squeeze(prices.to(self.device)) + self.num_users + self.num_items + self.num_cats

        scores = self.model.test_decode(self.output, users, items, cats, prices)

        return scores, num_positive


class PUPRecommender(BasePUPRecommender):

    def __init__(self, flags_obj, workspace, cm):

        super(PUPRecommender, self).__init__(flags_obj, workspace, cm)
        self.price_level = flags_obj.num_prices
    
    def set_gm(self, flags_obj):

        self.gm = utils.GraphManager(flags_obj)
    
    def set_pup_model(self):
    
        self.model = model.PUP(self.feature_size, self.embedding_size, self.dropout, self.alpha, self.split_dim)
    
    def get_dataloader_generator(self):

        return data.PriceLevelFactorizationDataLoaderGenerator(self.datafile_prefix, self.price_level)


class PUPRankRecommender(BasePUPRecommender):

    def __init__(self, flags_obj, workspace, cm):

        super(PUPRankRecommender, self).__init__(flags_obj, workspace, cm)
        self.price_level = flags_obj.num_prices
    
    def set_gm(self, flags_obj):

        self.gm = utils.RankGraphManager(flags_obj)
    
    def set_pup_model(self):
    
        self.model = model.PUP(self.feature_size, self.embedding_size, self.dropout, self.alpha, self.split_dim)
    
    def get_dataloader_generator(self):

        return data.RankPriceLevelFactorizationDataLoaderGenerator(self.datafile_prefix, self.price_level)


class PUPMinusCRecommender(BasePUPRecommender):

    def __init__(self, flags_obj, workspace, cm):

        super(PUPMinusCRecommender, self).__init__(flags_obj, workspace, cm)
    
    def set_gm(self, flags_obj):

        self.gm = utils.MinusGraphManager(flags_obj)
    
    def set_feature_size(self, flags_obj):

        self.feature_size = flags_obj.num_users + flags_obj.num_items + flags_obj.num_prices
    
    def set_pup_model(self):

        self.model = model.PUPMinusC(self.feature_size, self.embedding_size, self.dropout, self.alpha, self.split_dim)
    
    def inference(self, sample):

        user, item_p, item_n, _, _, price_p, price_n = sample
        user = user.to(self.device)
        item_p = item_p.to(self.device) + self.num_users
        item_n = item_n.to(self.device) + self.num_users
        price_p = price_p.to(self.device) + self.num_users + self.num_items
        price_n = price_n.to(self.device) + self.num_users + self.num_items
        p_score, n_score = self.model(self.gm.feature, self.gm.adj, user, item_p, item_n, price_p, price_n)

        return p_score, n_score
    
    def test_inference(self, sample):

        users, items, _, prices, num_positive = sample
        users = torch.squeeze(users.to(self.device))
        items = torch.squeeze(items.to(self.device)) + self.num_users
        prices = torch.squeeze(prices.to(self.device)) + self.num_users + self.num_items

        scores = self.model.test_decode(self.output, users, items, prices)

        return scores, num_positive


class PUPMinusPRecommender(BasePUPRecommender):

    def __init__(self, flags_obj, workspace, cm):

        super(PUPMinusPRecommender, self).__init__(flags_obj, workspace, cm)
    
    def set_gm(self, flags_obj):

        self.gm = utils.MinusGraphManager(flags_obj)
    
    def set_feature_size(self, flags_obj):

        self.feature_size = flags_obj.num_users + flags_obj.num_items + flags_obj.num_cats
    
    def set_pup_model(self):

        self.model = model.PUPMinusP(self.feature_size, self.embedding_size, self.dropout, self.alpha, self.split_dim)
    
    def inference(self, sample):

        user, item_p, item_n, cat_p, cat_n, _, _ = sample
        user = user.to(self.device)
        item_p = item_p.to(self.device) + self.num_users
        item_n = item_n.to(self.device) + self.num_users
        cat_p = cat_p.to(self.device) + self.num_users + self.num_items
        cat_n = cat_n.to(self.device) + self.num_users + self.num_items
        p_score, n_score = self.model(self.gm.feature, self.gm.adj, user, item_p, item_n, cat_p, cat_n)

        return p_score, n_score
    
    def test_inference(self, sample):

        users, items, cats, _, num_positive = sample
        users = torch.squeeze(users.to(self.device))
        items = torch.squeeze(items.to(self.device)) + self.num_users
        cats = torch.squeeze(cats.to(self.device)) + self.num_users + self.num_items

        scores = self.model.test_decode(self.output, users, items, cats)

        return scores, num_positive


class PUPMinusCPRecommender(BasePUPRecommender):

    def __init__(self, flags_obj, workspace, cm):

        super(PUPMinusCPRecommender, self).__init__(flags_obj, workspace, cm)
    
    def set_gm(self, flags_obj):

        self.gm = utils.MinusGraphManager(flags_obj)
    
    def set_feature_size(self, flags_obj):

        self.feature_size = flags_obj.num_users + flags_obj.num_items
    
    def set_pup_model(self):

        self.model = model.PUPMinusCP(self.feature_size, self.embedding_size, self.dropout, self.alpha, self.split_dim)
    
    def inference(self, sample):

        user, item_p, item_n, _, _, _, _ = sample
        user = user.to(self.device)
        item_p = item_p.to(self.device) + self.num_users
        item_n = item_n.to(self.device) + self.num_users
        p_score, n_score = self.model(self.gm.feature, self.gm.adj, user, item_p, item_n)

        return p_score, n_score
    
    def test_inference(self, sample):

        users, items, _, _, num_positive = sample
        users = torch.squeeze(users.to(self.device))
        items = torch.squeeze(items.to(self.device)) + self.num_users

        scores = self.model.test_decode(self.output, users, items)

        return scores, num_positive
