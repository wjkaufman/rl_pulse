import unittest
from rl_pulse import spin_simulation as ss
import numpy as np


class SpinSimulationTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a 4 spin-1/2 system.
        """
        self.N = 4
        self.dim = 2**4
    
    def test_mykron(self):
        self.assertTrue(np.array_equal(
            ss.mykron(ss.z, ss.z),
            np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]]) / 4))
        self.assertTrue(np.array_equal(
            ss.mykron(ss.x, ss.x),
            np.array([[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0]]) / 4))
        self.assertTrue(np.array_equal(
            ss.mykron(ss.z, ss.z, ss.z),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1]]) / 8))
    
    # def test():
    #     pass


if __name__ == '__main__':
    unittest.main()
