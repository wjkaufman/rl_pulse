import copy
import numpy as np
import spin_simulation as ss

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec, BoundedArraySpec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

class SpinSystemDiscreteEnv(py_environment.PyEnvironment):
    """A spin-1/2 system in a magnetic field.
    
    The actions are a discrete set of pulses that are applied to the spin
    system. This is analogous to the average Hamiltonian theory framework
    for Hamiltonian engineering (see link below).
    
    https://link.aps.org/doi/10.1103/PhysRevLett.20.180
    
    """
    
    def __init__(self, N, dim, coupling, delta, H_target, X, Y,\
            type='discrete', delay=5e-6, pulse_width=0, delay_after: bool=False):
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
        self.discount = 0.99
        
        self.delay = delay
        self.pulse_width = pulse_width
        self.delay_after = delay_after
        self.randomize = True
        
        self.make_actions()
    
    def action_spec(self):
        return BoundedArraySpec((5,), np.int32, minimum=0, maximum=1)
    
    def observation_spec(self):
        return BoundedArraySpec((None, 5,), np.int32, minimum=0, maximum=1)
        
    def _reset(self):
        '''Resets the environment by setting all propagators to the identity
        and setting t=0
        '''
        if self.randomize:
            _, self.Hint = ss.getAllH(self.N, self.dim, \
                self.coupling, self.delta)
            self.make_actions()
        if self.delay_after:
            self.Uexp = ss.getPropagator(self.Hint, self.delay)
            self.Utarget = ss.getPropagator(self.H_target, self.delay)
        else:
            self.Uexp = np.eye(self.dim, dtype="complex128")
            self.Utarget = np.copy(self.Uexp)
        # initialize time
        self.time = 0
        if self.delay_after:
            self.time += self.delay
        # for network training, define the "state" (sequence of actions)
        self.state = np.zeros((32, 5), dtype="float32")
        
        return ts.restart(self.state)
    
    def _step(self, action: np.ndarray):
        '''Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        
        Arguments:
            action: An ndarray with the corresponding action
        '''
        if self._current_time_step.is_last():
            return self._reset()
        
        # TODO change below when doing finite pulse widths/errors
        ind = int(np.nonzero(action)[0])
        self.Uexp = self.action_unitaries[ind] @ self.Uexp
        self.Utarget = ss.getPropagator(self.H_target, self.action_times[ind])
        self.time += self.action_times[ind]
        self.state[self.state_ind,:] = action
        self.state_ind += 1
        
        step_type = ts.StepType.MID
        if self.is_done():
            step_type = ts.StepType.LAST
        elif self.state_ind == 0:
            step_type = ts.StepType.FIRST
        
        reward = self.reward()
        
        return ts.TimeStep(step_type, reward, self.discount, self.state)
    
    # TODO write get_state and set_state methods
    
    def make_actions(self):
        '''Make a discrete number of propagators so that I'm not re-calculating
        the propagators over and over again.
        
        To simplify calculations, define each action as a pulse (or no pulse)
        followed by a delay
        '''
        Udelay = spla.expm(-1j*(self.Hint*self.delay))
        Ux = spla.expm(-1j*(self.X*np.pi/2))
        Uxbar = spla.expm(-1j*(self.X*-np.pi/2))
        Uy = spla.expm(-1j*(self.Y*np.pi/2))
        Uybar = spla.expm(-1j*(self.Y*-np.pi/2))
        if self.delay_after:
            Ux = Udelay @ Ux
            Uxbar = Udelay @ Uxbar
            Uy = Udelay @ Uy
            Uybar = Udelay @ Uybar
        self.unitaries = [Ux, Uxbar, Uy, Uybar, Udelay]
        self.times = [self.pulse_width, self.pulse_width, self.pulse_width, \
            self.pulse_width, self.delay]
    
    def copy(self):
        '''Return a copy of the environment
        '''
        return SpinSystemDiscreteEnv(self.N, self.dim, self.coupling, \
            self.delta, self.H_target, self.X, self.Y, type=self.type, \
            delay=self.delay, delay_after=self.delay_after)
    
    def reward(self):
        return -1.0 * (self.t > 1e-6) * \
            np.log10((1-ss.fidelity(self.Utarget,self.Uexp))+1e-100)
    
    def is_done(self):
        '''Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        TODO modify this when I move on from constrained (4-pulse) sequences
        '''
        return self.state_ind >= np.size(self.state, 0)
