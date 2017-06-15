# import modules
import pandas as pd
import numpy as np
from FPMC import FPMC
from sklearn.metrics import roc_curve

def main(fName, uSamples = 10, seed = 43):
    """
    Main fucntion to test properties of the prediction model
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
    ordersGroup = pd.read_pickle('../data/orders.pcl').groupby('user_id')
    items = pd.read_pickle('../data/items.pcl')

    print 'Test the model \n'

    mean_f1_total = 0.0
    fpmc_f1_total = 0.0

    for tUser in testUsers:
        print '\nUser {:d}'.format(tUser)
        userOrders = ordersGroup.get_group(tUser)
        testOrder = userOrders.query("eval_set != 'prior'")
        trainOrders = userOrders.query("eval_set == 'prior'")
        trainOrderIds = trainOrders.index.values

        if testOrder.eval_set.values == 'test':
            continue
        else:
            testOrderId = testOrder.index.values

        prevOrderNumber = testOrder.order_number.values - 1
        prevOrderId = userOrders.query('order_number == @prevOrderNumber').index.values
        prevBasket = items.query('order_id == @prevOrderId').product_id.values
        testBasket = items.query('order_id == @testOrderId').product_id.values

        ordIds = userOrders.index.values[:-1]

        userItems = items.query('order_id in @trainOrderIds')

        # find full basket
        fullBasket = userItems.groupby('product_id').count()
        fullBasket.rename(columns={'order_id': 'quantity'}, inplace=True)
        #fullBasket = fullBasket.merge(products, left_index=True, right_index=True).drop(['aisle_id', 'department_id'],
        #                                                                                axis=1)

        # calculate mean size of basket
        size = np.round(userItems.groupby('order_id').count().mean()).values

        for k, v in fullBasket.iterrows():
            fullBasket.loc[k, 'prob'] = FM.getProbability(basket=prevBasket, item=k, user=tUser)
            #fullBasket.loc[k, 'wasOrdered'] = k in testBasket

        # FPMC model
        recommendation = fullBasket.sort_values('prob', ascending=False).index.values[:int(size)]
        [pr, recall, fpmc_f1] = f1Score(testBasket, recommendation)
        fpmc_f1_total += fpmc_f1

        # Most popular model'
        recommendation = fullBasket.sort_values('quantity', ascending=False).index.values[:int(size)]
        [pr, recall, mean_f1] = f1Score(testBasket, recommendation)
        mean_f1_total += mean_f1

    print 'FPMC score {:f}'.format(fpmc_f1_total)
    print 'Mean score {:f}'.format(mean_f1_total)


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
    fName = 'fullModel-13June-3.pcl'
    main(fName,1000)

