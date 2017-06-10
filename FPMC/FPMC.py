"""
Implementation of Factoratized Personalized Markov Chains for Next-Basket Prediction by
S. Rendle, C. Freudenthaler and L. Schmidt-Thieme
"""

import numpy as np
import dill


class FPMC():

    def __init__(self, users=100, items=100, k=8, std=1.0):
        """
        Init instance of the class
        :param users: number of users
        :param items: number of items
        :param k: dimensionality of decomposition matrix
        """
        # counter of iterations
        self.iteration = 0

        # decomposition dimensionality
        self._kui = k
        self._kil = k

        # hyperparameters of SGD
        self._alpha = 0.1  # descend rate

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
        self._VIL = np.random.normal(0,self._std,[self.itemNumber,self._kui])
        self._VLI = np.random.normal(0,self._std,[self.itemNumber,self._kui])


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
        with open(fName, 'rb') as input:
            obj = dill.load(input)

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

    def setLearningRate(self,a):
        """
        Set learning rate
        :param a: learning rate 
        :return: 
        """
        self._alpha = a



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
            return 0

        deltaMean = 0.0
        for step in np.arange(nSteps):
            j = -1
            while (j==-1):
                guess = np.random.choice(self.itemNumber)
                if ~(newBasket == guess).sum():
                    j = guess

            delta = -1

            try:
                delta = 1.0 - self._sigma(iProb,self.getProbability(user,j,oldBasket))

                if ~np.isfinite(delta):
                    print 'delta is not finite'
                    print 'Item probability ',iProb
                    print 'Random item probability ',self.getProbability(user,j,oldBasket)
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
                        break

                    self._VIL[i,f] = self._VIL[i,f] + self._alpha*(delta*eta-self._lil*self._VIL[i,f])
                    self._VIL[j,f] = self._VIL[j,f] + self._alpha*(-delta*eta-self._lil*self._VIL[j,f])
                    self._VLI[oldBasket,f] = self._VLI[oldBasket,f] +\
                        self._alpha*(delta*(self._VIL[i,f]-self._VIL[j,f])/len(oldBasket)-self._lli*self._VLI[oldBasket,f])

                deltaMean += delta

            except Exception as ins:
                print ins
                print 'User', user, ', item ', i

        return deltaMean/nSteps

    @staticmethod
    def _sigma(x1,x2):
        return 1.0/(1+np.exp(-(x1-x2)))

