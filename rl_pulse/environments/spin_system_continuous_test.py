import unittest
import qutip as qt
import numpy as np
from rl_pulse.environments import spin_system_continuous


class SpinSystemContinuouseEnvTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a 4 spin-1/2 system.
        """
        N = 4  # 4-spin system
        chemical_shifts = np.array([50, -50, 100, -100])
        Hcs = sum(
            [qt.tensor(
                [qt.identity(2)]*i
                + [chemical_shifts[i] * qt.sigmaz()]
                + [qt.identity(2)]*(N-i-1)
            ) for i in range(N)]
        )
        dipolar_matrix = np.array([[0, 100, 50, 25],
                                   [0, 0, 100, 50],
                                   [0, 0, 0, 100],
                                   [0, 0, 0, 0]])
        Hdip = sum([
            dipolar_matrix[i, j] * (
                2 * qt.tensor(
                    [qt.identity(2)]*i
                    + [qt.sigmaz()]
                    + [qt.identity(2)]*(j-i-1)
                    + [qt.sigmaz()]
                    + [qt.identity(2)]*(N-j-1)
                )
                - qt.tensor(
                    [qt.identity(2)]*i
                    + [qt.sigmax()]
                    + [qt.identity(2)]*(j-i-1)
                    + [qt.sigmax()]
                    + [qt.identity(2)]*(N-j-1)
                )
                - qt.tensor(
                    [qt.identity(2)]*i
                    + [qt.sigmay()]
                    + [qt.identity(2)]*(j-i-1)
                    + [qt.sigmay()]
                    + [qt.identity(2)]*(N-j-1)
                )
            )
            for i in range(N) for j in range(i+1, N)
        ])
        
        Hsys = Hcs + Hdip
        X = qt.tensor([qt.sigmax()]*N)
        Y = qt.tensor([qt.sigmay()]*N)
        # Z = qt.tensor([qt.sigmaz()]*N)
        Hcontrols = [50e3 * X, 50e3 * Y]
        target = qt.propagator(X, np.pi/4)
        self.env = spin_system_continuous.SpinSystemContinuousEnv(
            Hsys,
            Hcontrols,
            target,
        )
    
    def test_step_types(self):
        step = self.env.reset()
        self.assertEqual(
            step.step_type,
            0,
            f'First step has step_type {step.step_type} (should be 0)')
        for action in [(0, 0), (.5, .5), (1, 1)]:
            step = self.env.step(action)
            self.assertEqual(
                step.step_type,
                1,
                f'Transition step has step_type {step.step_type}'
                + ' (should be 1)')
        # TODO fix below, need to perform steps until end of episode
        # step = self.env.step(0)
        # self.assertEqual(
        #     step.step_type,
        #     2,
        #     f'Last step has step_type {step.step_type} (should be 2)')
    
    # TODO test hard pulse rewards


if __name__ == '__main__':
    unittest.main()
