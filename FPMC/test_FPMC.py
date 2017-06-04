import unittest

from FPMC import FPMC
from numpy import testing as npt
import numpy as np
import os


class TestFPMC(unittest.TestCase):
    def test_save(self):
        """
        Test of save and load function
        :return: 
        """

        testFileName = 'testSave.pcl'

        obj = FPMC(300, 400)
        obj.save(testFileName)

        obj2 = FPMC(500, 450)
        obj2.load(testFileName)



        os.remove(testFileName)

        npt.assert_array_equal(obj._VUI, obj2._VUI)
        npt.assert_array_equal(obj._VIU, obj2._VIU)
        npt.assert_array_equal(obj._VIL, obj2._VIL)
        npt.assert_array_equal(obj._VLI, obj2._VLI)
        self.assertEqual(obj.userNumber,obj2.userNumber)
        self.assertEqual(obj.itemNumber,obj2.itemNumber)



    def test___init__(self):

        users1 = np.random.randint(100,1000)
        items1 = np.random.randint(1000,10000)
        obj1 = FPMC(users1, items1)
        print obj1.userNumber


        users2 = np.random.randint(100,1000)
        items2 = np.random.randint(1000,10000)
        obj2 = FPMC(users2, items2)

        print obj1.userNumber
        print obj2.userNumber

        # check properties value
        self.assertEqual(obj1.userNumber,users1)
        self.assertEqual(obj1.itemNumber,items1)

        # check properties value
        self.assertEqual(obj2.userNumber,users2)
        self.assertEqual(obj2.itemNumber,items2)



if __name__ == '__main__':
    unittest.main()