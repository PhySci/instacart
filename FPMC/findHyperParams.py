import numpy as np
import pandas as pd
from FPMC import FPMC as FM
import itertools
import matplotlib.pyplot as plt


itemsNumber = 49688


orders = pd.read_pickle('../data/orders.pcl')
ordersGroup = orders.groupby('user_id')
itemsGroup = pd.read_pickle('../data/items.pcl').groupby('order_id')

#learningRateList = [0.1, 0.05, 0.01]
#userNumberList = [100, 1000, 10000]

aRateList = [1e-7, 1e-9, 0]
userNumberList = [50000]

cycles = 20

for a, u in itertools.product(aRateList,userNumberList):
    print "Learning rate ",a,"user ", u

    obj = FM(users=u+1, items=itemsNumber+1, k=8)
    obj.setLearningRate(0.1, dynamic=True, dg=a)
    obj.setNormalization(0.1)

    logArr = np.empty((cycles*u,1), float)

    for ind in np.arange(1,cycles*u):
        user_orders = ordersGroup.get_group(ind)
        if user_orders.shape[0] < 3:
            break

        order_number = np.random.randint(1, user_orders.shape[0]-2)
        c = user_orders.iloc[order_number:(order_number + 2), :].index.values # id of train order (basket)
        basket = itemsGroup.get_group(c[1]).product_id.values
        prev_basket = itemsGroup.get_group(c[0]).product_id.values

        if len(prev_basket) == 0:
            print 'Empty basket is found'
            continue

        delta = np.abs(obj.SGD(u, basket, prev_basket, 10))
        logArr[ind - 1, 0] = delta

        #if (ind % 100 == 0):
        #    print 'Step ', int(ind),'. User is',u, '. Delta is ',

    np.savetxt('learning_log_'+str(a)+'_'+str(u)+'.csv', logArr, delimiter=',')
    plt.plot(logArr)


