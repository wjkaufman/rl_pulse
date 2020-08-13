import copy
import numpy as np
from scipy import linalg
import rl_pulse.spin_simulation as ss

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class SpinSystemDiscreteEnv(py_environment.PyEnvironment):
    """A spin-1/2 system in a magnetic field.
    
    The actions are a discrete set of pulses that are applied to the spin
    system. This is analogous to the average Hamiltonian theory framework
    for Hamiltonian engineering [^1].
    
    The observations made of the environment are the propagators of both
    the experimental ("actual") and target propagators.
    
    [^1]: https://link.aps.org/doi/10.1103/PhysRevLett.20.180
    
    """
    
    def __init__(
            self,
            N,
            dim,
            coupling,
            delta,
            H_target,
            delay=5e-6,
            pulse_width=0,
            delay_after=False,
            sparse_reward=False,
            episode_length=5):
        '''Initialize a new Environment object
        
        Arguments:
            delay_after: bool. Should there be a delay after every pulse/delay?
        '''
        
        super(SpinSystemDiscreteEnv, self).__init__()
        
        self.N = N
        self.dim = dim
        self.coupling = coupling
        self.delta = delta
        self.H_target = H_target
        
        (self.X, self.Y, self.Z) = ss.get_total_spin(N, dim)
        
        self.Hint = None
        self.Uexp = None
        self.Utarget = None
        self.time = 0
        
        self.ind = 0
        self.episode_length = episode_length
        self.action_unitaries = None
        self.action_times = None
        self.sparse_reward = sparse_reward
        self.discount = np.array(0.99, dtype="float32")
        
        self.delay = delay
        self.pulse_width = pulse_width
        self.delay_after = delay_after
        self.randomize = True
    
    def action_spec(self):
        return array_spec.BoundedArraySpec((), np.int32,
                                           minimum=0, maximum=4)
    
    def observation_spec(self):
        return array_spec.ArraySpec((self.dim, self.dim, 4), np.float32)
        
    def _reset(self):
        '''Resets the environment by setting all propagators to the identity
        and setting t=0
        '''
        if self.randomize:
            _, self.Hint = ss.get_H(self.N, self.dim,
                                    self.coupling, self.delta)
            self.make_actions()
        self.time = 0
        if self.delay_after:
            self.Uexp = ss.get_propagator(self.Hint, self.delay)
            self.Utarget = ss.get_propagator(self.H_target, self.delay)
            self.time += self.delay
        else:
            self.Uexp = np.eye(self.dim, dtype="complex128")
            self.Utarget = np.copy(self.Uexp)
        
        # for network training, define the "state" (sequence of actions)
        self.ind = 0
        
        return ts.restart(self.get_observation())
    
    def get_observation(self):
        """Return an observation from Uexp and Utarget"""
        obs = np.stack([self.Uexp.real, self.Uexp.imag,
                        self.Utarget.real, self.Utarget.imag],
                       axis=-1).astype(np.float32)
        return obs
    
    # TODO change get_state and set_state if it needs to be full copy of
    # environment
    def get_state(self):
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)
    
    def set_state(self, time_step: ts.TimeStep):
        self._current_time_step = time_step
        # TODO get other information from time_step
        # NOT below...
        # self.state = time_step.observation
    
    def _step(self, action: int):
        '''Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        
        Arguments:
            action: An ndarray with the corresponding action
        '''
        if self._current_time_step.is_last():
            return self._reset()
        
        self.Uexp = self.action_unitaries[action] @ self.Uexp
        self.Utarget = ss.get_propagator(self.H_target,
                                         self.action_times[action]) @ \
            self.Utarget
        self.time += self.action_times[action]
        
        step_type = ts.StepType.MID
        if self.is_done():
            step_type = ts.StepType.LAST
        
        r = self.reward(sparse_reward=self.sparse_reward,
                        reward_every=6)
        
        self.ind += 1
        
        return ts.TimeStep(step_type, np.array(r, dtype="float32"),
                           self.discount, self.get_observation())
    
    # TODO write get_state and set_state methods
    
    def make_actions(self):
        '''Make a discrete number of propagators so that I'm not re-calculating
        the propagators over and over again.
        
        To simplify calculations, define each action as a pulse (or no pulse)
        followed by a delay
        '''
        Udelay = linalg.expm(-1j*(self.Hint*self.delay))
        Ux = linalg.expm(-1j*(self.X*np.pi/2 + self.Hint*self.pulse_width))
        Uxbar = linalg.expm(-1j*(self.X*-np.pi/2 + self.Hint*self.pulse_width))
        Uy = linalg.expm(-1j*(self.Y*np.pi/2 + self.Hint*self.pulse_width))
        Uybar = linalg.expm(-1j*(self.Y*-np.pi/2 + self.Hint*self.pulse_width))
        if self.delay_after:
            Ux = Udelay @ Ux
            Uxbar = Udelay @ Uxbar
            Uy = Udelay @ Uy
            Uybar = Udelay @ Uybar
        self.action_unitaries = [Ux, Uxbar, Uy, Uybar, Udelay]
        if self.delay_after:
            self.action_times = [self.pulse_width + self.delay] * 4 + \
                                [self.delay]
        else:
            self.action_times = [self.pulse_width] * 4 + [self.delay]
    
    def copy(self):
        '''Return a copy of the environment
        '''
        return SpinSystemDiscreteEnv(self.N, self.dim, self.coupling,
                                     self.delta, self.H_target, self.X, self.Y,
                                     type=self.type, delay=self.delay,
                                     delay_after=self.delay_after)
    
    def reward(
            self,
            sparse_reward: bool = False,
            reward_every=1):
        """Get the reward for the current pulse sequence.
        
        Arguments:
            sparse_reward: If true, only return a reward at the end of the
                episode.
            reward_every: How often to give non-zero rewards. If the rewards
                are dense (`sparse_reward == False`) then rewards will be
                given every `reward_every` steps.
        
        """
        if sparse_reward and not self.is_done():
            return 0
        else:
            if ((self.ind > 0 and (self.ind + 2) % reward_every == 0)
                    or self.is_done()):
                r = (-1.0
                     * np.log10((1 - ss.fidelity(self.Utarget,
                                                 self.Uexp))
                                + 1e-100))
            else:
                r = 0
            return r
    
    def is_done(self):
        '''Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        '''
        return self.ind >= self.episode_length - 1
