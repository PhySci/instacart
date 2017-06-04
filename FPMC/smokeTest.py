from FPMC import FPMC as FM
import numpy as np
import pickle

def main():
    obj = FM()
    print obj.getProbability(50, 50, np.array([30, 60]))
    for i in np.arange(2):
        obj.descend(50,np.array([30, 60]),np.array([30, 80]))
    print 'Here'
    print obj.getProbability(50, 50, np.array([1]))

    obj.save('test.pcl')
    del obj

    with open('test.pcl', 'rb') as input:
        obj2 = pickle.load(input)
    print obj2.getProbability(50, 50, np.array([1]))

if __name__ == '__main__':
    main()
