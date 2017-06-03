from FPMC import FPMC as FM
import numpy as np

def main():
    obj = FM()
    print obj.getProbability(50, 50, np.array([10, 20, 30, 40, 50, 60, 70, 80]))
    print 'Here'



if __name__ == '__main__':
    main()
