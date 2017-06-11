"""
Here I want to test speed of query procedure
"""
import pandas as pd
import numpy as np
from time import time

#orders = pd.read_csv('../data/orders.csv',index_col = 'order_id',
#                     usecols = ['order_id','user_id','eval_set','order_number','days_since_prior_order'])
# orders.to_pickle('orders.pcl')

orders = pd.read_pickle('orders.pcl')

usecols = ['order_id','product_id']
#items = pd.concat([pd.read_csv('../data/order_products__train.csv',usecols = usecols),
#                   pd.read_csv('../data/order_products__prior.csv',usecols = usecols)])
#items.to_pickle('items.pcl')
items = pd.read_pickle('items.pcl')
itemsGroup = items.groupby('order_id')

#items.set_index(['order_id'],inplace=True)

maxUsers = 206290 # amount of users
testUsers = 100 # amount of test queries
seed = np.random.RandomState(seed = 42)
seed2 = np.random.RandomState(seed = 41)

userList = seed.randint(1, maxUsers,testUsers)

t = time()
for tUser in userList:
    user_orders = orders.query('user_id == @tUser')
    if user_orders.shape[0] < 3:
        break

    order_number = seed2.randint(1, user_orders.shape[0] - 2)  # order_number of order
    c = user_orders.iloc[order_number:(order_number + 2), :].index.values  # id of train and previous order (basket)
    basket = itemsGroup.get_group(c[0]).product_id.values
    prev_basket = itemsGroup.get_group(c[1]).product_id.values

print time()-t


t = time()
for tUser in userList:
    user_orders = orders.query('user_id == @tUser')
    if user_orders.shape[0] < 3:
        break
    order_number = seed2.randint(1, user_orders.shape[0] - 2)  # order_number of order
    c = user_orders.iloc[order_number:(order_number + 2), :].index.values  # id of train order (basket)
    g = items.query('order_id in @c').groupby('order_id')  # get all item from the train order

    basket = g.get_group(c[0]).product_id.values
    prev_basket = g.get_group(c[1]).product_id.values

print time()-t