import pandas as pd
import json

prefix = './data/'

# read csv file
action_log = pd.read_csv(prefix + 'action_log.csv')

# drop the 1st column
action_log = action_log.drop(labels=['Unnamed: 0'], axis=1)

# rename
action_log = action_log.rename(columns={'business_id': 'item_id'})

# extract items
items = action_log['item_id'].drop_duplicates()

# map to index & save to json file
item_index = {}
for i in range(items.count()):
    item_index[str(items.iloc[i])] = i
with open(prefix + 'item_index.json', 'w') as f:
    f.write(json.dumps(item_index))
print('finish generating item index!')

# extract users
users = action_log['user_id'].drop_duplicates()

# map to index & save to json file
user_index = {}
for i in range(users.count()):
    user_index[str(users.iloc[i])] = i
with open(prefix + 'user_index.json', 'w') as f:
    f.write(json.dumps(user_index))
print('finish generating user index!')


# invert index & save to json file
item_index_invert = {str(v): k for k, v in item_index.items()}
user_index_invert = {str(v): k for k, v in user_index.items()}
with open(prefix + 'item_index_invert.json', 'w') as f:
    f.write(json.dumps(item_index_invert))
with open(prefix + 'user_index_invert.json', 'w') as f:
    f.write(json.dumps(user_index_invert))
print('finish generating user & item invert index!')

# extract category
cats = action_log['categories'].drop_duplicates()
cats.reset_index(drop=True, inplace=True)

cats_list = []
for i in range(cats.count()):
    cats_list.append(cats[i])
cats_list = sorted(list(set(cats_list)))

# map to index
cat_index = {}
for i, cat in enumerate(cats_list):
    if cat in cat_index:
        print('error!')
    cat_index[cat] = i

# save to json file
with open(prefix + 'cat_index.json', 'w') as f:
    f.write(json.dumps(cat_index))
print('finish generating cat index!')

# map item to category
item_cat = action_log[['item_id', 'categories']].drop_duplicates()

item_cat_d = {}
for index, row in item_cat.iterrows():
    item_id = str(row['item_id'])
    cat = str(row['categories'])
    if item_id in item_cat_d:
        print('error!')
    item_cat_d[item_id] = cat

with open(prefix + 'item_cat.json', 'w') as f:
    f.write(json.dumps(item_cat_d))

# extract item price
item_p = action_log[['item_id', 'price']].drop_duplicates()

item_lux = {}
for index, row in item_p.iterrows():
    item_id = str(row['item_id'])
    price = row['price']
    if item_id in item_lux:
        print('error!')
    item_lux[item_id] = int(price)

with open(prefix + 'item_lux.json', 'w') as f:
    f.write(json.dumps(item_lux))
print('finish generating item lux!')
