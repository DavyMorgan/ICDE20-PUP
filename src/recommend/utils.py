#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import os
import datetime
import setproctitle
from absl import logging

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from visdom import Visdom

import config.const as const_util
import data
import recommender


class ContextManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name
        self.output = flags_obj.output
        self.workspace = flags_obj.workspace
    
    def set_recommender(self, flags_obj, workspace, cm):

        if flags_obj.model == 'MF':
            return recommender.MFRecommender(flags_obj, workspace, cm)      
        elif flags_obj.model == 'PUP':
            return recommender.PUPRecommender(flags_obj, workspace, cm)       
        elif flags_obj.model == 'PUP-C':
            return recommender.PUPMinusCRecommender(flags_obj, workspace, cm)
        elif flags_obj.model == 'PUP-P':
            return recommender.PUPMinusPRecommender(flags_obj, workspace, cm)
        elif flags_obj.model == 'PUP-CP':
            return recommender.PUPMinusCPRecommender(flags_obj, workspace, cm)
        elif flags_obj.model =='PUPRANK':
            return recommender.PUPRankRecommender(flags_obj, workspace, cm)
    
    def set_device(self, flags_obj):

        if not flags_obj.use_gpu:

            return torch.device('cpu')
        
        else:

            return torch.device('cuda:{}'.format(flags_obj.gpu_id))
    
    def set_default_ui(self):

        self.set_workspace()
        self.set_process_name()
        self.set_logging()
    
    def set_test_ui(self):

        self.set_process_name()
        self.set_test_logging()
    
    def set_workspace(self):

        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name = self.name + '_' + date_time
        self.workspace = os.path.join(self.output, dir_name)
        os.mkdir(self.workspace)
    
    def set_process_name(self):

        setproctitle.setproctitle(self.name + '@zhengyu')
    
    def set_logging(self):

        self.log_path = os.path.join(self.workspace, 'log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.name + '.log', self.log_path)
    
    def set_test_logging(self):

        self.log_path = os.path.join(self.workspace, 'test_log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.name + '.log', self.log_path)


class VizManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name
        self.port = flags_obj.port
        self.set_visdom()
    
    def set_visdom(self):

        self.viz = Visdom(port=self.port, env=self.name)
    
    def show_basic_info(self, flags_obj):

        basic = self.viz.text('Basic Information:')
        self.viz.text('Name: {}'.format(flags_obj.name), win=basic, append=True)
        self.viz.text('Model: {}'.format(flags_obj.model), win=basic, append=True)
        self.viz.text('Dataset: {}'.format(flags_obj.dataset), win=basic, append=True)
        self.viz.text('Embedding Size: {}'.format(flags_obj.embedding_size), win=basic, append=True)
        self.viz.text('Initial lr: {}'.format(flags_obj.lr), win=basic, append=True)
        self.viz.text('Batch Size: {}'.format(flags_obj.batch_size), win=basic, append=True)

        self.basic = basic
    
    def show_test_info(self, flags_obj):

        test = self.viz.text('Test Information:')
        self.viz.text('Test Mode: {}'.format(flags_obj.mode), win=test, append=True)
        self.viz.text('Workspace: {}'.format(flags_obj.workspace), win=test, append=True)

        self.test = test
    
    def update_line(self, title, epoch, loss):

        if epoch == 0:

            setattr(self, title, self.viz.line([loss], [epoch], opts=dict(title=title)))
        
        else:

            self.viz.line([loss], [epoch], win=getattr(self, title), update='append')
    
    def show_result(self, result):

        self.viz.text('-----Results-----', win=self.test, append=True)

        for i, k in enumerate(result['topk']):
            
            self.viz.text('topk: {}'.format(k), win=self.test, append=True)
            self.viz.text('Recall: {}'.format(result['recall'][i]), win=self.test, append=True)
            self.viz.text('NDCG: {}'.format(result['ndcg'][i]), win=self.test, append=True)
        
        self.viz.text('-----------------', win=self.test, append=True)


class ResourceManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name
        self.datafile_prefix = flags_obj.datafile_prefix
        self.all_items = np.arange(flags_obj.num_items, dtype=np.int32)
        self.load_positive_sample(flags_obj)
        self.load_cats(flags_obj)
        self.load_prices(flags_obj)
        self.num_workers = flags_obj.num_workers
        self.num_users = flags_obj.num_users
        self.topk = flags_obj.topk
    
    def load_positive_sample(self, flags_obj):

        with open(os.path.join(flags_obj.datafile_prefix, const_util.test_positive)) as f:

            self.test_positive = json.loads(f.read())
        
        with open(os.path.join(flags_obj.datafile_prefix, const_util.positive)) as f:

            self.positive = json.loads(f.read())
    
    def load_cats(self, flags_obj):

        self.cats = np.load(os.path.join(flags_obj.datafile_prefix, const_util.cats))
    
    def load_prices(self, flags_obj):

        if flags_obj.dataset == 'yelp':
            self.prices = np.load(os.path.join(flags_obj.datafile_prefix, const_util.prices))
        elif flags_obj.dataset == 'beibei':
            if flags_obj.model != 'PUPRANK':
                self.prices = np.load(os.path.join(flags_obj.datafile_prefix, str(flags_obj.num_prices) + '_' + const_util.prices))
            else:
                self.prices = np.load(os.path.join(flags_obj.datafile_prefix, str(flags_obj.num_prices) + '_rank_' + const_util.prices))
    
    def get_test_dataloader(self):

        return DataLoader(data.TestDataset(self), batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False)


class BaseGraphManager(object):

    def __init__(self, flags_obj):

        self.num_nodes = flags_obj.num_users + flags_obj.num_items + flags_obj.num_cats + flags_obj.num_prices
    
    def transfer_data(self, device):

        self.feature = self.feature.to(device)
        self.adj = self.adj.to(device)
    
    def generate_id_feature(self):

        i = torch.cat((torch.arange(self.num_nodes, dtype=torch.int64), torch.arange(self.num_nodes, dtype=torch.int64)), 0)
        i = i.reshape(2, -1)
        v = torch.ones(self.num_nodes)
        self.feature = torch.sparse.FloatTensor(i, v, torch.Size([self.num_nodes, self.num_nodes]))
    
    def generate_adj(self, flags_obj):

        train_positive, item_cat, item_index, cat_index, item_lux = self.load_data(flags_obj)

        row, col = self.generate_coo_row_col(flags_obj, train_positive, item_cat, item_index, cat_index, item_lux)

        self.generate_adj_from_coo_row_col(row, col)
    
    def load_data(self, flags_obj):

        with open(os.path.join(flags_obj.datafile_prefix, const_util.train_positive), 'r') as f:
            train_positive = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.item_cat), 'r') as f:
            item_cat = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.item_index), 'r') as f:
            item_index = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.cat_index), 'r') as f:
            cat_index = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.item_lux), 'r') as f:
            item_lux = json.loads(f.read())
        
        return train_positive, item_cat, item_index, cat_index, item_lux
    
    def generate_coo_row_col(self, flags_obj, train_positive, item_cat, item_index, cat_index, item_lux):

        count = 0
        for user, item in train_positive.items():
            count = count + len(item)
        count = count + 2 * flags_obj.num_items
        row = np.zeros(count, dtype=np.int32)
        col = np.zeros(count, dtype=np.int32)
        cursor = 0
        for user, item in train_positive.items():
            row[cursor: cursor + len(item)] = int(user)
            col[cursor: cursor + len(item)] = np.array(item) + flags_obj.num_users
            cursor = cursor + len(item)

        for item_id, cat_id in item_cat.items():
            r = item_index[item_id] + flags_obj.num_users
            c1 = cat_index[cat_id] + flags_obj.num_users + flags_obj.num_items
            if flags_obj.dataset == 'beibei':
                num_prices = flags_obj.num_prices
                lux = int(item_lux[item_id] * num_prices)
                if lux >= num_prices:
                    lux = num_prices - 1
            elif flags_obj.dataset == 'yelp':
                lux = int(item_lux[item_id]) - 1
            c2 = flags_obj.num_users + flags_obj.num_items + flags_obj.num_cats + lux

            row[cursor: cursor + 2] = r
            col[cursor] = c1
            col[cursor + 1] = c2
            cursor = cursor + 2
        
        return row, col
    
    def gennerate_adj_from_coo_row_col(self, row, col):

        pass


class GraphManager(BaseGraphManager):

    def __init__(self, flags_obj):

        super(GraphManager, self).__init__(flags_obj)
    
    def generate_adj_from_coo_row_col(self, row, col):

        adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        self.adj = adj
    
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


class RankGraphManager(GraphManager):

    def __init__(self, flags_obj):

        super(RankGraphManager, self).__init__(flags_obj)
    
    def load_data(self, flags_obj):

        with open(os.path.join(flags_obj.datafile_prefix, const_util.train_positive), 'r') as f:
            train_positive = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.item_cat), 'r') as f:
            item_cat = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.item_index), 'r') as f:
            item_index = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.cat_index), 'r') as f:
            cat_index = json.loads(f.read())
        with open(os.path.join(flags_obj.datafile_prefix, const_util.item_lux_rank), 'r') as f:
            item_lux = json.loads(f.read())
        
        return train_positive, item_cat, item_index, cat_index, item_lux


class MinusGraphManager(GraphManager):

    def __init__(self, flags_obj):

        super(MinusGraphManager, self).__init__(flags_obj)
        self.correct_num_nodes(flags_obj)
    
    def correct_num_nodes(self, flags_obj):

        if flags_obj.model == 'PUP-C':
            self.num_nodes = self.num_nodes - flags_obj.num_cats
        elif flags_obj.model == 'PUP-P':
            self.num_nodes = self.num_nodes - flags_obj.num_prices
        elif flags_obj.model == 'PUP-CP':
            self.num_nodes = self.num_nodes - flags_obj.num_cats - flags_obj.num_prices
    
    def generate_coo_row_col(self, flags_obj, train_positive, item_cat, item_index, cat_index, item_lux):

        count = 0
        for user, item in train_positive.items():
            count = count + len(item)
        count = count + 2 * flags_obj.num_items
        if flags_obj.model == 'PUP-C' or flags_obj.model == 'PUP-P':
            count = count - flags_obj.num_items
        elif flags_obj.model == 'PUP-CP':
            count = count - 2 * flags_obj.num_items
        row = np.zeros(count, dtype=np.int32)
        col = np.zeros(count, dtype=np.int32)
        cursor = 0
        for user, item in train_positive.items():
            row[cursor: cursor + len(item)] = int(user)
            col[cursor: cursor + len(item)] = np.array(item) + flags_obj.num_users
            cursor = cursor + len(item)
        
        if flags_obj.model == 'PUP-CP':
            return row, col

        for item_id, cat_id in item_cat.items():
            r = item_index[item_id] + flags_obj.num_users
            if flags_obj.dataset == 'beibei':
                num_prices = flags_obj.num_prices
                lux = int(item_lux[item_id] * num_prices)
                if lux == num_prices:
                    lux = num_prices - 1
            elif flags_obj.dataseet == 'yelp':
                lux = int(item_lux[item_id]) - 1

            row[cursor] = r
            if flags_obj.model == 'PUP-C':
                col[cursor] = flags_obj.num_users + flags_obj.num_items + lux
            elif flags_obj.model == 'PUP-P':                
                col[cursor] = flags_obj.num_users + flags_obj.num_items + cat_index[cat_id]
            cursor = cursor + 1
        
        return row, col
    