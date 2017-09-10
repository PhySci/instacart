"""
Implementation of Factoratized Personalized Markov Chains for Next-Basket Prediction by
S. Rendle, C. Freudenthaler and L. Schmidt-Thieme
"""

import numpy as np
import dill
import logging


class FPMC():

    def __init__(self, users=100, items=100, ku=8, ki = 8, std=1.0):
        """
        Init instance of the class
        :param users: number of users
        :param items: number of items
        :param k: dimensionality of decomposition matrix
        """

        # set logging info
        logging.basicConfig(filename='FPMC.log',format='%(asctime)s %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # set numpy properties
        np.seterr(over='raise')

        # counter of iterations
        self.iteration = 0

        # decomposition dimensionality
        self._kui = ku
        self._kil = ki

        # hyperparameters of SGD
        self._alpha = 0.1  # descend rate
        self._alpha0 = 0.1
        self._dynamic_rate = False
        self._dg = 0

        # normalization coefficients
        self._lui = 0.1
        self._liu = 0.1
        self._lil = 0.1
        self._lli = 0.1

        # set object properties
        self.userNumber = users
        self.itemNumber = items
        self._std = float(std)

        # Fill out matrix
        self._VUI = np.random.normal(0,self._std,[self.userNumber,self._kui])
        self._VIU = np.random.normal(0,self._std,[self.itemNumber,self._kui])
        self._VIL = np.random.normal(0,self._std,[self.itemNumber,self._kil])
        self._VLI = np.random.normal(0,self._std,[self.itemNumber,self._kil])
        self.logger.info('The instance was created.')


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

    def SGD(self, user, newBasket, oldBasket, nSteps=1):
        """
        Make one descendent step
        :param user: user_id
        :param newBasket: array of items from new basket
        :param oldBasket: array of items from previous basket
        :return: none
        """

        #for step in np.arange(nSteps+1):
        i = np.random.choice(newBasket)
        return self._descend(user, i, newBasket, oldBasket, nSteps)

    def addOrder(self, user, newBasket, oldBasket, iterations = 1000):
        """
        Add all items from newBasket to FPMC model
        :param user: 
        :param oldBasket: 
        :param newBasket: 
        :param iterations: 
        :return: 
        """

        # loop over all items in newBasket
        for item in newBasket:
            self._descend(user, item, newBasket, oldBasket, iterations)

    def save(self,fName):
        """
        Save the object to a file
        :param fName: file name
        :return: 
        """
        with open(fName, 'wb') as output:
            dill.dump(self, output)

    def load(self,fName):
        """
        Load object from file
        :return: 
        """
        try:
            with open(fName, 'rb') as input:
                obj = dill.load(input)
        except EOFError as ins:
            print 'File is absent.'
            print ins
        else:
            for key in self.__dict__.keys():
                setattr(self,key,getattr(obj,key))

    def setNormalization(self,a):
        """
        Set normalization coefficients
        :param a: 
        :return: 
        """
        self._lli = a
        self._lil = a
        self._liu = a
        self._lui = a

    def setLearningRate(self,a,dynamic = False, dg = 0):
        """
        Set learning rate
        :param a: learning rate
        :param dynamic: set adaptive learning rate
        :param df: rate of decreasing of learning rate
        :return: 
        """
        self._alpha = a
        self._alpha0 = a
        self._dynamic_rate = dynamic
        self._dg = dg


    def _descend(self, user, i, newBasket, oldBasket, nSteps = 1):
        """
        One step of descend
        :param user       - user id
        :param i          - one item from new basket   
        :param oldBasket  - previous order (array)
        :param newBasket  - new order
        :param nSteps     - number of SGD steps
        """
        self.iteration = self.iteration +1

        # probability for the current item i
        iProb = self.getProbability(user,i,oldBasket)

        if ~np.isfinite(iProb):
            print 'Non-finite probability for item', i,'. User ',user
            self.logger.warning('Non-finite probability. User %d, item %d', user, i)
            return 0

        deltaMean = 0.0
        for step in np.arange(nSteps):

            j = -1
            while (j==-1):
                guess = np.random.choice(self.itemNumber)
                if ~(newBasket == guess).sum():
                    j = guess

            delta = -1

            jProb = self.getProbability(user,j,oldBasket)
            delta = 1.0 - self._sigma(iProb,jProb)

            if ~np.isfinite(delta):
                print 'delta is not finite'
                print 'Item probability ',iProb
                print 'Random item probability ',jProb
                self.logger.warning('Delta value is not finite. User %d, item %d, item probability %f, item random probability %f',
                                    user, i, iProb, jProb)
                return -1

            #for f in np.arange(self._kui):
            self._VUI[user,:] = self._VUI[user,:] + self._alpha*\
                 (delta*(self._VIU[i,:]-self._VIU[j,:])-self._lui*self._VUI[user,:])

            self._VIU[i,:] = self._VIU[i,:] + self._alpha*\
                  (delta*self._VUI[user,:]-self._liu*self._VIU[i,:])

            self._VIU[j,:] = self._VIU[j,:] + self._alpha*\
                  (-delta*self._VUI[user,:]-self._liu*self._VIU[j,:])

            for f in np.arange(self._kil):
                eta = np.sum(self._VLI[oldBasket, f]) / len(oldBasket)

                if ~np.isfinite(eta):
                    print 'eta is not finite'
                    self.logger.warning(
                        'Eta value is not finite. User %d, item %d, item probability is %f, item random probability is %f, delts is %f',
                        user, i, iProb, iProb, delta)
                    break

                self._VIL[i,f] = self._VIL[i,f] + self._alpha*(delta*eta-self._lil*self._VIL[i,f])
                self._VIL[j,f] = self._VIL[j,f] + self._alpha*(-delta*eta-self._lil*self._VIL[j,f])
                self._VLI[oldBasket,f] = self._VLI[oldBasket,f] +\
                    self._alpha*(delta*(self._VIL[i,f]-self._VIL[j,f])/len(oldBasket)-self._lli*self._VLI[oldBasket,f])

            deltaMean += delta

        if (self.iteration % 10 == 0 and self._dynamic_rate):
            self._adjustLearningRate()

        return deltaMean/nSteps

    def _sigma(self,x1,x2):
        """
        Calculate value of sigmoid function of difference x1 and x2 
        :param x1: probability of first item
        :param x2: probability of second item
        :return: 
        """
        try:
            dx = x1-x2
            res =  1.0/(1+np.exp(-dx))
        except Exception as ins:
            print ins
            if dx <0:
                res = 0
            else:
                res = 1
            self.logger.warning('Sigma calculation error. dx is %f', dx)
        finally:
            return res

    def _adjustLearningRate(self):
        self._alpha = self._alpha0/(1+self._alpha0*self._dg*self.iteration)

