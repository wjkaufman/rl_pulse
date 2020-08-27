import unittest
from rl_pulse import spin_simulation as ss
import spin_sys_discrete


class SpinSystemDiscreteEnvTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a 4 spin-1/2 system.
        """
        (X, Y, Z) = ss.get_total_spin(4, 2**4)
        H_target = ss.get_H_WHH_0(X, Y, Z, delta=500)
        self.env = spin_sys_discrete.SpinSystemDiscreteEnv(
            N=4, dim=2**4, coupling=1e3, delta=500, H_target=H_target,
            delay_after=True, episode_length=5
        )
        
    def test_step_types(self):
        step = self.env.reset()
        self.assertEqual(
            step.step_type,
            0,
            f'First step has step_type {step.step_type} (should be 0)')
        for action in [1, 2, 4, 3]:
            step = self.env.step(1)
            self.assertEqual(
                step.step_type,
                1,
                f'Transition step has step_type {step.step_type}'
                + ' (should be 1)')
        step = self.env.step(0)
        self.assertEqual(
            step.step_type,
            2,
            f'Last step has step_type {step.step_type} (should be 2)')
        # TODO continue here
    
    def test_sparse_rewards(self):
        self.env.sparse_reward = True
        step = self.env.reset()
        self.assertEqual(step.reward, 0, 'First reward should be zero')
        for action in [1, 2, 4, 3]:
            step = self.env.step(action)
            self.assertEqual(
                step.reward,
                0,
                'Reward should be zero with sparse_reward == True')
        step = self.env.step(0)
        self.assertGreater(
            step.reward,
            0,
            'Reward should be non-zero'
        )
    
    def test_dense_rewards(self):
        self.env.sparse_reward = False
        step = self.env.reset()
        self.assertEqual(step.reward, 0, 'First reward should be zero')
    

if __name__ == '__main__':
    unittest.main()
