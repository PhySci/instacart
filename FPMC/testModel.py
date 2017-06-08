# import modules
import pandas as pd
import numpy as np
from FPMC import FPMC
from sklearn.metrics import roc_curve


def main(fName, uSamples = 10, seed = 43):
    """
    Main fucntion to test propeties of the prediction model
    :param fName:    name of file with the model
    :param uSamples: number of users for testing
    :param seed:     seed of random generator
    :return: 
    """
    FM = FPMC()
    FM.load(fName)

    # create ramdom list of test users
    rnd = np.random.RandomState(seed)
    testUsers = rnd.randint(1,FM.userNumber+1,uSamples)

    # load data
    print 'Load data'
    products = pd.read_csv('../data/products.csv', index_col='product_id')
    orders = pd.read_csv('../data/orders.csv', index_col='order_id',
                         usecols=['order_id', 'user_id', 'eval_set', 'order_number', 'days_since_prior_order'])
    usecols = ['order_id', 'product_id']
    items = pd.concat([pd.read_csv('../data/order_products__train.csv', usecols=usecols),
                       pd.read_csv('../data/order_products__prior.csv', usecols=usecols)])

    print 'Test the model'

    for tUser in testUsers:
        tOrders = orders.query('user_id == @tUser')
        testOrder = tOrders.query("eval_set != 'prior'")

        if testOrder.eval_set.values == 'test':
            print 'Test user'
            continue
        else:
            testOrderId = testOrder.index.values

        # train the model
        prevOrderId = tOrders.query('order_number == 1').index.values
        prevBasket = items.query('order_id == @prevOrderId').product_id.values

        #for newOrderId in tOrders.index.values[1:-1]:
        #    # print 'Order id is', newOrderId
        #    newBasket = items.query('order_id == @newOrderId').product_id.values
        #    FM.addOrder(tUser, newBasket, prevBasket, iterations=1e4)
        #    prevBasket = newBasket

        testBasket = items.query('order_id == @testOrderId').product_id.values

        ordIds = tOrders.index.values[:-1]
        fullBasket = items.query('order_id in @ordIds').drop_duplicates('product_id')
        fullBasket = fullBasket.merge(products, left_on='product_id', right_index=True).drop(
            ['aisle_id', 'department_id'],
            axis=1)

        for k, v in fullBasket.iterrows():
            fullBasket.loc[k, 'prob'] = FM.getProbability(basket=prevBasket, item=v.product_id, user=tUser)
            fullBasket.loc[k, 'wasOrdered'] = v.product_id in testBasket

        fullBasket.sort_values('wasOrdered', ascending=True, inplace=True)
        print roc_curve(fullBasket.wasOrdered, fullBasket.prob)

    return

def f1Score(y_true,y_pred):
    """
    Return F1 score
    :param y_true: true values (array of item_id)
    :param y_pred: predicted values (array of item_id)
    :return: [precession, recall, f1]
    """
    intersection = np.intersect1d(y_true, y_pred).size
    precession = intersection / float(y_pred.shape[0])
    recall = intersection / float(y_true.shape[0])
    try:
        f1 = 2 * precession * recall / (precession + recall)
    except ZeroDivisionError:
        f1 = 0
    return [precession, recall, f1]

    pass
    return [0.0,0.0,0.0]




if __name__ == '__main__':
    fName = 'testModel-8June.pcl'
    main(fName)

