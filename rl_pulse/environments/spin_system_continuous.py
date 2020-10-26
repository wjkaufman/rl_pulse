import copy
import numpy as np
import qutip as qt

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class SpinSystemContinuousEnv(py_environment.PyEnvironment):
    """A spin-1/2 system in a magnetic field.
    
    The goal in this environment is to implement a target propagator
    by controlling amplitudes of certain terms in the Hamiltonian.
    
    The actions are a continuous set of control amplitudes that are
    applied to the spin system. Specifically, the control amplitudes are
    magnetic field strengths and phases applied to the spin system.
    
    The observations made of the environment are the propagators of both
    the experimental ("actual") and target propagators.
    Though the observations could also be the sequence of actions already
    applied to the system.
    """
    
    def __init__(
            self,
            Hsys,
            Hcontrols,
            target,
            initial_state=None,
            dt=1e-7,
            T=1e-5,
            num_steps=100,
            discount_factor=0.99
            ):
        """Initialize a new Environment object
        
        Arguments:
            Hsys (Qobj): Time-independent system Hamiltonian.
            Hcontrols (array of Qobj): List of control Hamiltonians. The
                control amplitudes are between -1 and 1, so the control
                Hamiltonians correspond to a "fully on" control field.
            target (Qobj): Target state or target unitary transformation.
            initial_state (Qobj): Initial state, for implementing a
                state-to-state transfer. Defaults to `None` for implementing
                a unitary transformation.
            dt (float): Time interval for each time step.
            T (float): Max episode time in seconds.
            discount_factor (float): Discount factor to calculate return.
        """
        
        super(SpinSystemContinuousEnv, self).__init__()
        
        self.Hsys = Hsys
        self.Hcontrols = Hcontrols
        self.target = target
        
        # initial_state may be `None` for unitary transformation
        self.initial_state = initial_state
        self.dt = dt
        self.T = T
        self.discount = np.array(discount_factor, dtype="float32")
        
        self._action_spec = array_spec.BoundedArraySpec(
            (len(self.Hcontrols),),
            np.float32,
            minimum=-1, maximum=1
        )
        self._observation_spec = array_spec.ArraySpec(
            (None, len(self.Hcontrols),),
            np.float32
        )
        
        self._reset()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
        
    def _reset(self):
        """Resets the environment by setting all propagators to the identity
        and setting t=0
        """
        self.propagator = qt.identity(self.Hsys.dims[0])
        self.t = 0
        self.actions = np.zeros((int(self.T/self.dt), len(self.Hcontrols)))
        self.index = 0
        
        return ts.restart(
            np.zeros((1, len(self.Hcontrols,)), dtype=np.float32)
        )
    
    def get_observation(self):
        """Return an observation from Uexp and Utarget. Used
        for non-sequential observations on the system (i.e.
        full observations of the system state).
        """
        if self.initial_state is not None:
            state = (self.propagator
                     * self.initial_state
                     * self.propagator.dag()).full()
        else:
            state = self.propagator
        target = self.target.full()
        obs = np.stack([state.real, state.imag,
                        target.real, target.imag],
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
    
    def _step(self, action):
        """Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        
        Arguments:
            action: An ndarray with spec according to `action_spec`.
        """
        if self._current_time_step.is_last():
            return self._reset()
        
        # propagate the system according to action
        H = (
            self.Hsys
            + sum([action[i] * Hc for i, Hc in enumerate(self.Hcontrols)]))
        U = qt.propagator(H, self.dt)
        self.propagator = U * self.propagator
        self.t += self.dt
        
        self.actions[self.index] = action
        self.index += 1
        
        step_type = ts.StepType.MID
        if self.is_done():
            step_type = ts.StepType.LAST
        
        r = self.reward()
        
        return ts.TimeStep(
            step_type,
            np.array(r, dtype=np.float32),
            self.discount,
            self.actions[:self.index]
        )
    
    def reward(self):
        """Get the reward for the current pulse sequence.
        """
        if self.initial_state is not None:
            # reward for state-to-state transfer
            fidelity = (
                self.propagator * self.initial_state * self.propagator.dag()
                * self.target
            ).tr() / self.target.shape[0]
        else:
            fidelity = ((self.propagator.dag() * self.target).tr()
                        / self.target.shape[0])
        r = -1.0 * np.log10(1 - fidelity + 1e-100)
        return np.abs(r)
    
    def is_done(self):
        """Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        """
        return self.t >= self.T
