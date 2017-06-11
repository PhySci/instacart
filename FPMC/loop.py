import numpy as np
import pandas as pd
from FPMC import FPMC as FM
from time import time

fName = 'testModel-10June7.pcl'
logFile = 'testModel-10June7.csv'

userNumber =  1000# 100 # 206290 - full set
itemsNumber = 49688

#orders = pd.read_csv('../data/orders.csv',index_col = 'order_id',
#                     usecols = ['order_id','user_id','eval_set','order_number','days_since_prior_order'])

#usecols = ['order_id','product_id']
#items = pd.concat([pd.read_csv('../data/order_products__train.csv',usecols = usecols),
#                   pd.read_csv('../data/order_products__prior.csv',usecols = usecols)])

orders = pd.read_pickle('orders.pcl')
itemsGroup = pd.read_pickle('items.pcl').groupby('order_id')

t = time()



obj = FM(users=userNumber+1, items=itemsNumber+1, k=8)
obj.setLearningRate(0.1)
obj.setNormalization(0.1)

print 'load the model'
#obj.load(fName)

print obj.iteration, ' has been done'

steps2save = 1000
logArr = np.empty((0,2), int)

for ind in np.arange(1,userNumber):
    user = int(ind % obj.userNumber)

    if user ==0:
        continue

    user_orders = orders.query('user_id == @user')
    if user_orders.shape[0] < 3:
        break

    nSteps = 1
    for step in np.arange(nSteps):
        delta = 0.0

        order_number = np.random.randint(1, user_orders.shape[0]-2)

        c = user_orders.iloc[order_number:(order_number + 2), :].index.values # id of train order (basket)
        #g = items.query('order_id in @c').groupby('order_id') # get all item from the train order

        #basket = g.get_group(c[0]).product_id.values
        #prev_basket = g.get_group(c[1]).product_id.values

        basket = itemsGroup.get_group(c[0]).product_id.values
        prev_basket = itemsGroup.get_group(c[1]).product_id.values

        if len(prev_basket) == 0:
            print 'Empty basket is found'
            continue

        d = np.abs(obj.SGD(user, basket, prev_basket, 10))
        logArr = np.vstack([logArr, np.array([obj.iteration, d])])
        delta += d

    if (ind % 100 == 0):
        print 'Step ', int(ind),'. User is',user, '. Delta is ',delta/nSteps

    if (ind % steps2save == 0):
         print ' Save the model.'
         obj.save(fName)

         with open(logFile, 'a') as f_handle:
             np.savetxt(f_handle, logArr, delimiter=',')
             logArr = np.empty((0, 2), int)


