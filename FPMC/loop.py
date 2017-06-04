import pandas as pd
import numpy as np
import pickle
from FPMC import FPMC as FM


fName = 'tesModel2.pcl'

orders = pd.read_csv('../data/orders.csv',index_col = 'order_id',
                     usecols = ['order_id','user_id','eval_set','order_number','days_since_prior_order'])

ordersId = orders.query("(user_id < 100) and (eval_set =='prior') and (order_number >1)").index.values

usecols = ['order_id','product_id']
items = pd.concat([pd.read_csv('../data/order_products__train.csv',usecols = usecols),
                   pd.read_csv('../data/order_products__prior.csv',usecols = usecols)])

#with open(fName, 'rb') as input:
#    self = pickle.load(input)

obj = FM()
obj.load(fName)


for ind in np.arange(1e5):
    #try:
    user = np.random.randint(1, 101)
    print 'Step ',ind,', user ', user
    user_orders = orders.query('user_id == @user')
    order_number = np.random.randint(1, user_orders.shape[0]-2)

    c = user_orders.iloc[order_number:(order_number + 2), :].index.values
    g = items.query('order_id in @c').groupby('order_id')

    basket = g.get_group(c[0]).product_id.values
    prev_basket = g.get_group(c[1]).product_id.values

    obj.descend(user, basket, prev_basket)

    if (ind % 1000 == 0):
         obj.save(fName)
         print 'Save'
    #except:
    #    print 'User', user, ', order_number ', order_number