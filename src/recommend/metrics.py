#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import logging

import torch
import numpy as np

import random


def calc_hit_recall_ndcg(scores, num_positive, topks, verbal):

    hit_recall = [0.0 for _ in topks]
    hit_ndcg = [0.0 for _ in topks]

    maxk = max(topks)
    top_scores, top_indices = torch.topk(scores, maxk)

    if verbal and random.random() < 0.001:

        logging.info('positive {} top10 {}'.format(scores[0], top_scores[:10]))

    for i, k in enumerate(topks):
        temp_top_indices = top_indices[:k]
        temp = min(k, num_positive)

        hit_positive = (temp_top_indices < num_positive).sum().item()

        index = np.arange(k)
        idcg = (1 / np.log(2 + np.arange(temp))).sum()
        dcg = (1 / np.log(2 + index[temp_top_indices.numpy() < num_positive])).sum()

        hit_recall[i] = hit_positive / temp
        hit_ndcg[i] = dcg / idcg

    return hit_recall, hit_ndcg