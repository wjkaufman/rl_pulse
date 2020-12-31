import unittest
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath('.'))
import alpha_zero as az


class AlphaZeroTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a 4 spin-1/2 system.
        """
        self.p = az.Policy()
        self.v = az.Value()
        sequence = torch.randint(5, (48,))
        self.states = [az.one_hot_encode(sequence[:i].float())
                       for i in range(48)]
        self.probs = F.softmax(torch.randn(48, 5), 1)
        self.values = torch.randn(48, 1)
    
    def test_packed(self):
        packed_states = az.pad_and_pack(self.states)
        output, _ = self.p(packed_states)
        output_individual = torch.cat([self.p(s.unsqueeze(0))[0]
                                       for s in self.states])
        norm = torch.norm(output - output_individual)
        self.assertAlmostEqual(float(norm), 0, 6)
        # value function
        output, _ = self.v(packed_states)
        output_individual = torch.cat([self.v(s.unsqueeze(0))[0]
                                       for s in self.states])
        print(output.shape, output_individual.shape)
        norm = torch.norm(output - output_individual)
        self.assertAlmostEqual(float(norm), 0, 6)
    
    def test_hidden_cell_states(self):
        packed_states = az.pad_and_pack(self.states)
        output, (h, c) = self.p(packed_states)
        output1, (h1, c1) = self.p(packed_states, h0=h, c0=c)
        doubled_states = [
            torch.cat([s, s])
            for s in self.states
        ]
        packed_doubles = az.pad_and_pack(doubled_states)
        output2, (h2, c2) = self.p(packed_doubles)
        norm_output = torch.norm(output1 - output2)
        norm_h = torch.norm(h1 - h2)
        norm_c = torch.norm(c1 - c2)
        self.assertAlmostEqual(float(norm_output), 0, 6)
        self.assertAlmostEqual(float(norm_h), 0, 6)
        self.assertAlmostEqual(float(norm_c), 0, 6)
        # TODO add test for value network too


if __name__ == '__main__':
    unittest.main()
