# encoding: utf-8

import numpy as np
import torch
import os
import logging
import time

from multiprocessing import Process

import utils

from tqdm import tqdm

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='generate_training_data.log', level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def generate_epoch_training_data(epoch, prefix, num_users, num_items,
                                 train_positive, positive, neg_sample_rate, total_sample, sales, alpha, cats, prices):

    users = np.zeros([total_sample], dtype=np.int32)
    items_p = np.zeros([total_sample], dtype=np.int32)
    items_n = np.zeros([total_sample], dtype=np.int32)

    temp = np.array([int(j) for j in range(num_items)], dtype=np.int32)

    cursor = 0
    for i in range(num_users):

        train_positive_i = train_positive[str(i)]
        positive_i = positive[str(i)]
        num_train_positive = len(train_positive_i)
        num_sample_positive = num_train_positive
        num_sample_negative = neg_sample_rate * num_sample_positive
        num_sample = num_sample_negative
        train_positive_i_np = np.array(train_positive_i, dtype=np.int32)
        positive_i_np = np.array(positive_i, dtype=np.int32)
        negative_i_np = temp[np.logical_not(np.isin(temp, positive_i_np))]

        prob = sales[negative_i_np]
        prob = prob ** alpha
        prob = prob / prob.sum()
        used_negative_i_np = np.random.choice(negative_i_np, num_sample_negative, replace=False, p=prob)

        users[cursor:cursor + num_sample] = i
        items_p[cursor:cursor + num_sample] = train_positive_i_np[:]
        items_n[cursor:cursor + num_sample] = used_negative_i_np[:]
        cursor = cursor + num_sample

    users_tensor = torch.LongTensor(users)
    items_p_tensor = torch.LongTensor(items_p)
    items_n_tensor = torch.LongTensor(items_n)
    cats_p_tensor = torch.LongTensor(cats[items_p])
    cats_n_tensor = torch.LongTensor(cats[items_n])
    prices_p_tensor = torch.LongTensor(prices[items_p])
    prices_n_tensor = torch.LongTensor(prices[items_n])

    path = prefix + 'epoch_' + str(epoch)
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(users_tensor, path + '/users.pth')
    torch.save(items_p_tensor, path + '/items_p.pth')
    torch.save(items_n_tensor, path + '/items_n.pth')
    torch.save(cats_p_tensor, path + '/cats_p.pth')
    torch.save(cats_n_tensor, path + '/cats_n.pth')
    torch.save(prices_p_tensor, path + '/prices_p.pth')
    torch.save(prices_n_tensor, path + '/prices_n.pth')


def generate_training_data(epochs, train_positive, positive, neg_sample_rate,
                           num_items, num_users, prefix, sales, alpha, cats, prices):

    total_sample = 0
    for i in range(num_users):

        pos = train_positive[str(i)]
        total_sample = total_sample + len(pos) * (1 + neg_sample_rate)

    path = prefix + 'train_data/'
    if not os.path.exists(path):
        os.mkdir(path)

    print('generating training data!')
    for epoch in tqdm(range(epochs)):

        generate_epoch_training_data(epoch, path, num_users, num_items, train_positive, positive,
                                     neg_sample_rate, total_sample, sales, alpha, cats, prices)

    return True


class Producer(Process):

    def __init__(self, epochs, total_sample, num_items, num_users, train_positive, positive,
                 neg_sample_rate, sales, alpha, path, cats, prices):
        super(Producer, self).__init__()
        self.epochs = epochs
        self.total_sample = total_sample
        self.num_items = num_items
        self.num_users = num_users
        self.train_positive = train_positive
        self.positive = positive
        self.neg_sample_rate = neg_sample_rate
        self.sales = sales
        self.alpha = alpha
        self.path = path
        self.cats = cats
        self.prices = prices

    def run(self):
        np.random.seed()
        for epoch in self.epochs:
            print('epoch {} start!'.format(epoch))
            start_time = time.time()
            generate_epoch_training_data(epoch, self.path, self.num_users, self.num_items, self.train_positive,
                                         self.positive, self.neg_sample_rate, self.total_sample,
                                         self.sales, self.alpha, self.cats, self.prices)
            print('epoch {} complete using {:.4f}s!'.format(epoch, time.time() - start_time))


def multi_generate_training_data(num_workers, epochs, train_positive, positive, neg_sample_rate,
                                 num_items, num_users, prefix, sales, alpha, cats, prices):
    total_sample = 0
    for i in range(num_users):
        pos = train_positive[str(i)]
        total_sample = total_sample + len(pos)

    path = prefix + 'train_data/'
    if not os.path.exists(path):
        os.mkdir(path)

    processes = []
    for i in range(num_workers):
        duty_epochs = [j for j in range(epochs) if j % num_workers == i]
        p = Producer(duty_epochs, total_sample, num_items, num_users, train_positive, positive,
                     neg_sample_rate, sales, alpha, path, cats, prices)
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    return True


if __name__ == '__main__':

    prefix = '/data3/zhengyu/price_yelp_restaurant_fm_bpr/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    bridge_prefix = './data/'

    item_index, item_index_invert, item_cat, cat_index, item_lux, user_index, user_index_invert \
        = utils.get_bridge(bridge_prefix)

    from_file = True
    cats, prices = utils.generate_cat_price(item_index_invert, item_cat, cat_index, item_lux, from_file, prefix)

    num_users = len(user_index)
    num_items = len(item_index)

    from_file = True
    multi_process_generating = True
    if multi_process_generating:
        num_workers = 20
        positive, train_positive, val_positive, test_positive \
            = utils.multi_split_train_val(num_workers, user_index_invert, item_index, prefix, bridge_prefix, from_file)
    else:
        positive, train_positive, val_positive, test_positive \
            = utils.split_train_val(user_index_invert, item_index, prefix, bridge_prefix, from_file)

    from_file = True
    sales = utils.compute_sales(prefix, train_positive, num_items, from_file)
    sales = sales.numpy()
    sales = sales + 1
    alpha = 0

    epochs = 200
    neg_sample_rate = 1

    generate_training = True
    multi_process_generating = True
    if generate_training and not multi_process_generating:
        success = generate_training_data(epochs, train_positive, positive, neg_sample_rate, num_items, num_users,
                                         prefix, sales, alpha, cats, prices)
        if not success:
            print('fail!')
            exit()
    elif generate_training and multi_process_generating:
        num_workers = 20
        success = multi_generate_training_data(num_workers, epochs, train_positive, positive, neg_sample_rate,
                                               num_items, num_users, prefix, sales, alpha, cats, prices)

        if not success:
            print('fail!')
            exit()
