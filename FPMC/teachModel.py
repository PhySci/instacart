import numpy as np
import pandas as pd
from FPMC import FPMC as FM
from time import time

fName = 'fullModel-16June-3.pcl'
logFile = 'fullModel-16June-3.csv'

userNumber =  206209 #- full set
itemsNumber = 49688

ordersGroup = pd.read_pickle('../data/orders.pcl').groupby('user_id')
itemsGroup  = pd.read_pickle('../data/items.pcl').groupby('order_id')



obj = FM(users=userNumber+1, items=itemsNumber+1, ku = 512, ki = 512)
obj.load(fName)
obj.setLearningRate(0.1, dynamic = True, dg = 1e-5)
obj.iteration = 1
obj.setNormalization(0.05)

print obj.iteration, ' has been done'

steps2save = 1000
steps2show = 100000
logArr = np.empty((0,2), int)

for ind in np.arange(1,500*userNumber):
    user = np.random.randint(1,obj.userNumber)

    user_orders = ordersGroup.get_group(user)
    if user_orders.shape[0] < 3:
        break

    delta = 0.0

    order_number = np.random.randint(1, user_orders.shape[0]-2)
    c = user_orders.iloc[order_number:(order_number + 2), :].index.values # id of train order (basket)
    basket = itemsGroup.get_group(c[1]).product_id.values
    prev_basket = itemsGroup.get_group(c[0]).product_id.values

    if len(prev_basket) == 0:
        print 'Empty basket is found'
        continue

    delta = np.abs(obj.SGD(user, basket, prev_basket, 1))
    logArr = np.vstack([logArr, np.array([obj.iteration, delta])])


    if (ind % steps2show == 0):
         print 'Step is {:d}, rate is {:1.5f}'.format(ind, obj._alpha)
         with open(logFile, 'a') as f_handle:
             np.savetxt(f_handle, logArr, delimiter=',')
             logArr = np.empty((0, 2), int)

    if (ind % steps2save == 0):
         print ' Save the model. Step is ', ind
         obj.save(fName)

