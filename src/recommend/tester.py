#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

import utils
import recommender
import metrics


class Tester(object):

    def __init__(self, flags_obj, trained_recommender, cm, vm):

        self.cm = cm
        self.vm = vm
        self.name = flags_obj.name
        self.model = flags_obj.model
        self.dataset = flags_obj.dataset
        self.workspace = cm.workspace
        self.set_recommender(flags_obj, trained_recommender, cm.workspace)
        self.set_rm(flags_obj)
        self.set_dataloader()
    
    def set_recommender(self, flags_obj, trained_recommender, workspace):

        pass
    
    def set_rm(self, flags_obj):

        self.rm = utils.ResourceManager(flags_obj)
    
    def set_dataloader(self):
 
        self.dataloader = self.rm.get_test_dataloader()
    
    def test(self):

        hit_recall = np.zeros(len(self.rm.topk), dtype=np.float64)
        hit_ndcg = np.zeros(len(self.rm.topk), dtype=np.float64)

        with torch.no_grad():

            self.recommender.prepare_test()

            for _, sample in enumerate(tqdm(self.dataloader)):
                
                scores, num_positive = self.recommender.test_inference(sample)
                scores = scores.to(torch.device('cpu'))

                hit_recall_u, hit_ndcg_u = metrics.calc_hit_recall_ndcg(scores, num_positive.item(), self.rm.topk, True)

                hit_recall = hit_recall + hit_recall_u
                hit_ndcg = hit_ndcg + hit_ndcg_u

        recall = hit_recall / self.rm.num_users
        ndcg = hit_ndcg / self.rm.num_users

        self.report(recall, ndcg)
    
    def report(self, recall, ndcg):

        metrics_path = os.path.join(self.workspace, 'metrics')
        if not os.path.exists(metrics_path):
            os.mkdir(metrics_path)
        
        result_path = os.path.join(metrics_path, 'basic.json')

        result = {
            'topk': self.rm.topk,
            'recall': recall.tolist(),
            'ndcg': ndcg.tolist(),
        }

        with open(result_path, 'w') as f:

            f.write(json.dumps(result))
        
        self.vm.show_result(result)


class InstantTester(Tester):

    def __init__(self, flags_obj, trained_recommender, cm, vm):

        super(InstantTester, self).__init__(flags_obj, trained_recommender, cm, vm)
        self.workspace = trained_recommender.workspace
    
    def set_recommender(self, flags_obj, trained_recommender, workspace):

        self.recommender = trained_recommender


class PostTester(Tester):

    def __init__(self, flags_obj, trained_recommender, cm, vm):

        super(PostTester, self).__init__(flags_obj, trained_recommender, cm, vm)
        self.prepare_user_study()
    
    def set_recommender(self, flags_obj, trained_recommender, workspace):

        self.recommender = self.cm.set_recommender(flags_obj, workspace, self.cm)
        
        self.recommender.transfer_model()
        self.recommender.load_ckpt()

    def prepare_user_study(self):

        self.user_study = []

    def get_low_high_cate_price(self, uid):

        items = self.rm.positive[str(uid)]
        cates = self.rm.cats[items]
        prices = self.rm.prices[items]

        df = pd.DataFrame({'cate': cates, 'price': prices})
        df = df.groupby('cate').mean().reset_index()

        df_min = df[df.price == df.price.min()].reset_index(drop=True)
        lo_cate = df_min['cate'][0]
        lo_price = df_min['price'][0]

        df_max = df[df.price == df.price.max()].reset_index(drop=True)
        hi_cate = df_max['cate'][0]
        hi_price = df_max['price'][0]

        return lo_cate, lo_price, hi_cate, hi_price

    def get_recommend_low_high_price(self, top_items, lo_cate, hi_cate):

        lo_items = top_items[self.rm.cats[top_items] == lo_cate]
        if len(lo_items) < 1:
            return False, None, None
        recommend_lo_price = self.rm.prices[lo_items].mean()

        hi_items = top_items[self.rm.cats[top_items] == hi_cate]
        if len(hi_items) < 1:
            return False, None, None
        recommend_hi_price = self.rm.prices[hi_items].mean()

        return True, recommend_lo_price, recommend_hi_price

    def update_user_study(self, sample, scores):

        users, items, _, _, _ = sample
        uid = users[0][0].item()
        lo_cate, lo_price, hi_cate, hi_price = self.get_low_high_cate_price(uid)

        _, top_indices = torch.topk(scores, 1000)
        top_items = items[0][top_indices].numpy()

        recommended, recommend_lo_price, recommend_hi_price = self.get_recommend_low_high_price(top_items, lo_cate, hi_cate)

        if recommended:
            self.user_study.append([uid, lo_cate, lo_price, recommend_lo_price, hi_cate, hi_price, recommend_hi_price])

    def report(self, recall, ndcg):

        metrics_path = os.path.join(self.workspace, 'metrics')
        if not os.path.exists(metrics_path):
            os.mkdir(metrics_path)
        
        result_path = os.path.join(metrics_path, 'user_study.npy')
        result = np.array(self.user_study)

        np.save(result_path, result)

        result_path = os.path.join(metrics_path, 'basic.json')

        result = {
            'topk': self.rm.topk,
            'recall': recall.tolist(),
            'ndcg': ndcg.tolist(),
        }

        with open(result_path, 'w') as f:

            f.write(json.dumps(result))
        
        self.vm.show_result(result)

    def test(self):

        hit_recall = np.zeros(len(self.rm.topk), dtype=np.float64)
        hit_ndcg = np.zeros(len(self.rm.topk), dtype=np.float64)

        with torch.no_grad():

            self.recommender.prepare_test()

            for _, sample in enumerate(tqdm(self.dataloader)):
                
                scores, num_positive = self.recommender.test_inference(sample)
                scores = scores.to(torch.device('cpu'))

                self.update_user_study(sample, scores)

                hit_recall_u, hit_ndcg_u = metrics.calc_hit_recall_ndcg(scores, num_positive.item(), self.rm.topk, True)

                hit_recall = hit_recall + hit_recall_u
                hit_ndcg = hit_ndcg + hit_ndcg_u

        recall = hit_recall / self.rm.num_users
        ndcg = hit_ndcg / self.rm.num_users

        self.report(recall, ndcg)
    