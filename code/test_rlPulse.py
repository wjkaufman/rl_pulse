import rlPulse as rlp
import numpy as np
import unittest

# TODO fill in unit tests here...

class ActionTestCase(unittest.TestCase):
    def setUp(self):
        self.a0 = np.zeros((1,3))
        self.a = np.array([[.2,.5,.7]])
        self.aLarge = np.array([[-1,5,2]])
    
    def test_action_encoding(self):
        self.assertEqual(rlp.getPhiFromAction(self.a0), 0, 'incorrect phi encoding')
        self.assertEqual(rlp.getRotFromAction(self.a0), 0, 'incorrect rot encoding')
        self.assertEqual(rlp.getTimeFromAction(self.a0), 0, 'incorrect phi encoding')
    
    def test_action_clipping(self):
        pass
        # check if clipping aLarge puts it in between 0 and 1

class MutateTestCase(unittest.TestCase):
    def setUp(self):
        self.mat = np.random.normal(size=(3,4))
        self.matCopy = np.copy(self.mat)
    
    def test_no_mutation(self):
        rlp.mutateMat(self.matCopy, 1, 0, 0, 0)
        self.assertTrue(np.array_equal(self.mat, self.matCopy))
