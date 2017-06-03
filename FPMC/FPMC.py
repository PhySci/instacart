"""
Implementation of Factoratized Personalized Markov Chains for Next-Basket Prediction by
S. Rendle, C. Freudenthaler and L. Schmidt-Thieme
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import linalg
#from scipy import linalg

class FPMC():

    # number of users
    _userNumber = 100

    # number of items
    _itemNumber = 100

    # decomposition matrix
    _VUI = []
    _VIU = []
    _VIL = []
    _VLI = []

    # decomposition dimensionality
    _kui = 8
    _kil = 8

    # standart deviation of initial normal distribution
    _sigma = 1.0

    # hyperparameters of SGD
    _alpha = 0.1 # descrent rate

    # penalty coefficients
    _lui = 0.1
    _liu = 0.1
    _lil = 0.1
    _lli = 0.1


    @classmethod
    def __init__(self, users= 100, items = 100, k = 8, sigma = 1):
        """
        Init instance of the cless
        :param users: number of users
        :param items: number of items
        :param k: dimensionality of decomposition matrix
        """

        # set object properties
        self._userNumber = users
        self._itemNumber = items
        self._kui = k
        self._kil = k
        self._sigma = float(sigma)

        # Fill out matrix
        self._VUI = np.random.normal(0,self._sigma,[self._userNumber,self._kui])
        self._VIU = np.random.normal(0,self._sigma,[self._itemNumber,self._kui])
        self._VIL = np.random.normal(0,self._sigma,[self._itemNumber,self._kui])
        self._VLI = np.random.normal(0,self._sigma,[self._itemNumber,self._kui])


    @classmethod
    def getProbability(self,user,item,basket):
        """
        Return probability of ordering "item" by "user" if previous order is "basket" 
        :param user: user id
        :param item: item id
        :param basket: array of items from previous order
        :return: probability
        """
        res = np.dot(self._VUI[user,:],self._VIU[item,:])

        # @TODO The loop is not so good. Try to replace it for numpy.tensordot or Einstein summation
        # For instance, duplicate firsts term and use it as a two matrix
        sm = 0
        for l in basket:
            sm = sm + np.dot(self._VIL[item,:],self._VLI[l,:])

        res = res + sm/len(basket)
        return res

    def descend(self,user,newBasket,oldBasket):
        """
        Make one descendent step
        :param user: user_id
        :param newBasket: array of items from new basket
        :param oldBasket: array of items from previous basket
        :return: none
        """

        # line 4
        i = np.random.choice(newBasket)

        #line 5
        j = -1
        while (j==-1):
            guess = np.random.choice(self._itemNumber)
            if ~(newBasket == guess).sum():
                j = guess

        #line 6
        delta = 1.0 - self._sigma*(self.getProbability(user,i,oldBasket)-self.getProbability(user,j,oldBasket))

        # line 7
        for f in np.arange(self._kui):
            # line 8
            self._VUI[user,f] = self._VUI[user,f] + self._alpha*\
                 (delta*(self._VIU[i,f]-self._VIU[j,f])-self._lui*self._VUI[user,f])

        pass


