import pandas as pd
import numpy as np
from FPMC import FPMC as FM


fName = 'tesModel3.pcl'
maxUser = 100 # 206290 - full set

orders = pd.read_csv('../data/orders.csv',index_col = 'order_id',
                     usecols = ['order_id','user_id','eval_set','order_number','days_since_prior_order'])


usecols = ['order_id','product_id']
items = pd.concat([pd.read_csv('../data/order_products__train.csv',usecols = usecols),
                   pd.read_csv('../data/order_products__prior.csv',usecols = usecols)])




obj = FM(maxUser+1,49689)
obj.load(fName)
print obj._alpha

for ind in np.arange(1e6):
    user = np.random.randint(1, maxUser+1)
    user_orders = orders.query('user_id == @user')
    order_number = np.random.randint(1, user_orders.shape[0]-2)

    c = user_orders.iloc[order_number:(order_number + 2), :].index.values
    g = items.query('order_id in @c').groupby('order_id')

    basket = g.get_group(c[0]).product_id.values
    prev_basket = g.get_group(c[1]).product_id.values

    delta = obj.SGD(user, basket, prev_basket)
    print 'Step ', int(ind), ', user ', user, ', delta is', delta

    if (ind % 1000 == 0):
         obj.save(fName)
         print 'Save'