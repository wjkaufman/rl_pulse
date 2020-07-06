import copy
import numpy as np
from scipy import linalg
import spin_simulation as ss

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class SpinSystemDiscreteEnv(py_environment.PyEnvironment):
    """A spin-1/2 system in a magnetic field.
    
    The actions are a discrete set of pulses that are applied to the spin
    system. This is analogous to the average Hamiltonian theory framework
    for Hamiltonian engineering (see link below).
    
    https://link.aps.org/doi/10.1103/PhysRevLett.20.180
    
    """
    
    def __init__(self, N, dim, coupling, delta, H_target, X, Y,
                 delay=5e-6, pulse_width=0, delay_after: bool = False):
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
        self.X = X
        self.Y = Y
        
        self.Hint = None
        self.Uexp = None
        self.Utarget = None
        self.time = 0
        
        self.state_ind = 0
        self.state = None
        self.action_unitaries = None
        self.action_times = None
        self.reward_last = 0.0
        self.discount = np.array(0.99, dtype="float32")
        
        self.delay = delay
        self.pulse_width = pulse_width
        self.delay_after = delay_after
        self.randomize = True
    
    def action_spec(self):
        return array_spec.BoundedArraySpec((), np.int32,
                                           minimum=0, maximum=4)
    
    def observation_spec(self):
        # TODO eventually want to explore variable-time inputs?
        return array_spec.BoundedArraySpec((32, 5,), np.int32,
                                           minimum=0, maximum=1)
        
    def _reset(self):
        '''Resets the environment by setting all propagators to the identity
        and setting t=0
        '''
        if self.randomize:
            _, self.Hint = ss.getAllH(self.N, self.dim,
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
        self.state = np.zeros((32, 5), dtype="int32")
        self.state_ind = 0
        
        return ts.restart(self.state)
    
    def get_state(self):
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)
    
    def set_state(self, time_step: ts.TimeStep):
        self._current_time_step = time_step
        self.state = time_step.observation
    
    def _step(self, action: int):
        '''Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        
        Arguments:
            action: An ndarray with the corresponding action
        '''
        if self._current_time_step.is_last():
            return self._reset()
        
        # TODO change below when doing finite pulse widths/errors
        ind = action
        self.Uexp = self.action_unitaries[ind] @ self.Uexp
        self.Utarget = ss.get_propagator(self.H_target, self.action_times[ind])
        self.time += self.action_times[ind]
        state_representation = np.zeros((5,), dtype=int)
        state_representation[ind] = 1
        self.state[self.state_ind, :] = state_representation
                
        step_type = ts.StepType.MID
        if self.is_done():
            step_type = ts.StepType.LAST
        elif self.state_ind == 0:
            step_type = ts.StepType.FIRST
            
        self.state_ind += 1
        reward = self.reward()
        r = reward - self.reward_last
        if self.action_times[ind] > 0:
            self.reward_last = 0.0
        else:
            self.reward_last = reward
        
        return ts.TimeStep(step_type, np.array(r, dtype="float32"),
                           self.discount, self.state)
    
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
        # TODO correct action_times for delay_after condition
        self.action_times = [self.pulse_width, self.pulse_width,
                             self.pulse_width, self.pulse_width, self.delay]
    
    def copy(self):
        '''Return a copy of the environment
        '''
        return SpinSystemDiscreteEnv(self.N, self.dim, self.coupling,
                                     self.delta, self.H_target, self.X, self.Y,
                                     type=self.type, delay=self.delay,
                                     delay_after=self.delay_after)
    
    def reward(self):
        r = -1.0 * (self.time > 1e-6) * \
            np.log10((1-ss.fidelity(self.Utarget, self.Uexp)) + 1e-100)
        if r < 3:
            r = 0.0
        elif r > 5:
            r *= 100.0
        return r
    
    def is_done(self):
        '''Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        TODO modify this when I move on from constrained (4-pulse) sequences
        '''
        return self.state_ind >= np.size(self.state, 0) - 1
