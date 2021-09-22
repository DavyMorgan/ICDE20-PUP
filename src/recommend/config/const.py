#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

train_data = 'train_data/'

train_positive = 'train_positive.json'
val_positive = 'val_positive.json'
test_positive = 'test_positive.json'
positive = 'positive.json'

cats = 'cats.npy'
prices = 'prices.npy'
item_cat = 'item_cat.json'
item_index = 'item_index.json'
cat_index = 'cat_index.json'
item_lux = 'item_lux.json'
item_lux_rank = 'item_lux_rank.json'

user = 'users.pth'
item_p = 'items_p.pth'
item_n = 'items_n.pth'
cat_p = 'cats_p.pth'
cat_n = 'cats_n.pth'
price_p = 'prices_p.pth'
price_n = 'prices_n.pth'

ckpt = 'ckpt/'
model = 'epoch_199.pth'

prices_absolute = 'prices_absolute.npy'
prices_rank = 'prices_rank.npy'
