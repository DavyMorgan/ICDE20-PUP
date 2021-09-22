# encoding: utf-8

import pandas as pd

import json

from tqdm import tqdm

prefix = '/data3/zhengyu/yelp/'

with open(prefix + 'categories.json', 'r') as f:
    cats = json.loads(f.read())

cats_slim = [cc for cc in cats if cc['parents'] == ['restaurants']]

business = []
with open(prefix + 'yelp_academic_dataset_business.json', 'r') as f:
    print('reading business file!')
    for bb in tqdm(f):
        business.append(json.loads(bb))

review = []
with open(prefix + 'yelp_academic_dataset_review.json', 'r') as f:
    print('reading review file!')
    for rr in tqdm(f):
        review.append(json.loads(rr))

print('transforming to DataFrames!')
business_pd = pd.DataFrame(business)
review_pd = pd.DataFrame(review)

business_filter_na = business_pd.dropna(subset=['business_id', 'categories', 'attributes'])

print('filtering price & category!')
business_filter_na_dict = business_filter_na.to_dict('records')
business_filter_price_dict = [bb for bb in business_filter_na_dict if 'RestaurantsPriceRange2' in bb['attributes']]
for i, bb in tqdm(enumerate(business_filter_price_dict)):
    price = bb['attributes']['RestaurantsPriceRange2']
    business_filter_price_dict[i]['price'] = price

business_filter_price_cat_dict = []
use_cats = set([cc['title'] for cc in cats_slim])
for i, bb in tqdm(enumerate(business_filter_price_dict)):
    business_cat = list(set(bb['categories'].replace(' ', '').split(',')).intersection(use_cats))
    if len(business_cat) == 1:
        bb['categories'] = business_cat[0]
        business_filter_price_cat_dict.append(bb)
business_filter_price = pd.DataFrame(business_filter_price_cat_dict)

review_pd_filter_price = review_pd.merge(business_filter_price, on='business_id')

print('filtering duplicating reviews, merged to the last review!')
review_pd_filter_price = review_pd_filter_price[['business_id', 'review_id', 'user_id', 'date']] \
                        .sort_values(by='date', ascending=False) \
                        .drop_duplicates(subset=['business_id', 'user_id'])

print('filtering items!')
items_count = review_pd_filter_price[['business_id', 'user_id']] \
            .groupby('business_id') \
            .count() \
            .reset_index() \
            .rename(index=str, columns={'user_id': 'count'})

items_count_filtered = items_count[items_count['count'] > 10]

review_pd_filter_price_item = review_pd_filter_price.merge(items_count_filtered, on='business_id')

print('filtering users!')
users_count = review_pd_filter_price_item[['user_id', 'business_id']] \
            .groupby('user_id') \
            .count() \
            .reset_index() \
            .rename(index=str, columns={'business_id': 'count'})

users_count_filtered = users_count[users_count['count'] > 10]

review_pd_filter_price_item_user = review_pd_filter_price_item.merge(users_count_filtered, on='user_id')

action_log = review_pd_filter_price_item_user[['business_id', 'user_id', 'review_id', 'date']] \
            .merge(business_filter_price[['business_id', 'price', 'categories']], on='business_id')

print('total interaction count: {}'.format(action_log['review_id'].count()))

items_count = action_log[['business_id', 'user_id']] \
            .groupby('business_id') \
            .count() \
            .reset_index() \
            .rename(index=str, columns={'user_id': 'count'})

print('items count: {}\t min count: {}\t max count: {}\t mean count: {}'
      .format(items_count['count'].count(),
              items_count['count'].min(),
              items_count['count'].max(),
              items_count['count'].mean()))

users_count = action_log[['user_id', 'business_id']] \
            .groupby('user_id') \
            .count() \
            .reset_index() \
            .rename(index=str, columns={'business_id': 'count'})

print('users count: {}\t min count: {}\t max count: {}\t mean count: {}'
      .format(users_count['count'].count(),
              users_count['count'].min(),
              users_count['count'].max(),
              users_count['count'].mean()))

print('writing to file!')
action_log.to_csv('./data/action_log.csv')
