import rlPulse as rlp
import numpy as np
import unittest

# TODO fill in unit tests here...

class ActionTestCase(unittest.TestCase):
    def setUp(self):
        self.a0 = rlp.Action(np.zeros((1,3)), 'continuous')
        self.a1 = rlp.Action(np.array([[.2,.5,.7]]), 'continuous')
        self.a2 = rlp.Action([0,0,1,0,0], 'discrete')
        self.aLarge = np.array([[-1,5,2]])
        
        self.state = np.array([[0,0,0,0,1], [1,0,0,0,0], [0,0,1,0,0]])
    
    def test_action_encoding(self):
        self.assertEqual(self.a0.getPhi(), 0, 'incorrect phi encoding')
        self.assertEqual(self.a0.getRot(), 0, 'incorrect rot encoding')
        # self.assertEqual(rlp.getTimeFromAction(self.a0), 0, 'incorrect time encoding')
        # discrete actions
        self.assertEqual(self.a2.getPhi(), np.pi/2, \
            'incorrect phi encoding for discrete action')
        self.assertEqual(self.a2.getRot(), np.pi/2, \
            'incorrect rot encoding for discrete action')
        
    
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
