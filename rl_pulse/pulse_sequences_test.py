import unittest
import sys
import os
sys.path.append(os.path.abspath('.'))
import pulse_sequences as ps
import qutip as qt
import numpy as np


class PulseSequencesTest(unittest.TestCase):
    
    def setUp(self):
        # TODO fill in setup here, test all the different methods I've written
        pass
    
    # TODO fill in more here


class PulseSequenceConfigTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a ps_config object and
        """
        N = 3
        Utarget = qt.tensor([qt.identity(2)] * N)
        ps_config = ps.PulseSequenceConfig(
            N=N,
            ensemble_size=3,
            sequence_length=48,
            Utarget=Utarget,
            pulse_width=0.005,
            delay=0.01,
            )
        
    def test_yxx48(self):
        self.ps_config.reset()
        for p in ps.yxx48:
            self.ps_config.apply(p)
        self.assertGreater(self.ps_config.value(), 2)
    
    def test_frame(self):
        self.ps_config.reset()
        self.assertEqual(self.ps_config.frame, np.eye(3))
        self.ps_config.apply(0)
        self.assertEqual(self.ps_config.frame, np.eye(3))
    
    def test_clone(self):
        # TODO fill in
        pass
    
    # TODO more here...


if __name__ == '__main__':
    unittest.main()
