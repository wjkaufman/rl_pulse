import copy
import numpy as np
import qutip as qt
import tensorflow as tf
from collections import namedtuple
from math import ceil


class SpinSystemContinuousEnv:
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
    
    TimeStep = namedtuple(
        'TimeStep',
        'step_type reward discount observation')
    
    def __init__(
            self,
            Hsys,
            Hcontrols,
            target,
            initial_state=None,
            dt=5e-7,
            T=5e-5,
            discount=0.99
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
        
        self.Hsys = Hsys
        self.Hcontrols = Hcontrols
        self.target = target
        
        # initial_state may be `None` for unitary transformation
        self.initial_state = initial_state
        self.dt = dt
        self.T = T
        self.discount = tf.constant(discount, shape=(1,), dtype=tf.float32)
        
        # TODO replace these with TensorSpecs
        # self._action_spec = array_spec.BoundedArraySpec(
        #     (len(self.Hcontrols),),
        #     np.float32,
        #     minimum=-1, maximum=1
        # )
        # self._observation_spec = array_spec.ArraySpec(
        #     (None, len(self.Hcontrols),),
        #     np.float32
        # )
        
        self.reset()
    
    # TODO uncomment when I redefine specs
    # def action_spec(self):
    #     return self._action_spec
    #
    # def observation_spec(self):
    #     return self._observation_spec
        
    def reset(self):
        """Resets the environment by setting all propagators to the identity
        and setting t=0
        """
        self.propagator = qt.identity(self.Hsys.dims[0])
        self.t = 0
        self.actions = np.zeros(
            (1, ceil(self.T/self.dt), len(self.Hcontrols)),
            dtype=np.float32)
        self.index = 0
        self.previous_reward = 0
        
        # return an initial timestep
        return self.TimeStep(
            tf.constant(0, shape=(1,), dtype=tf.int32),
            self.reward(),
            tf.constant(1, shape=(1,), dtype=tf.float32),
            tf.zeros((1, 1, len(self.Hcontrols)), dtype=tf.float32))
    
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
        obs = tf.stack([state.real, state.imag,
                        target.real, target.imag],
                       axis=-1).astype(tf.float32)
        return obs
    
    # TODO change get_state and set_state if it needs to be full copy of
    # environment
    def get_state(self):
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)
    
    def set_state(self, time_step):
        self._current_time_step = time_step
        # TODO get other information from time_step
    
    def step(self, action):
        """Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        
        Arguments:
            action: An ndarray with spec according to `action_spec`.
        """
        if self.is_done():
            return self.reset()
        
        action = tf.squeeze(action).numpy()
        
        # propagate the system according to action
        H = (
            self.Hsys
            + sum([action[i] * Hc for i, Hc in enumerate(self.Hcontrols)]))
        U = qt.propagator(H, self.dt)
        self.propagator = U * self.propagator
        self.t += self.dt
        
        self.actions[0, self.index, :] = action
        self.index += 1
        
        step_type = 1  # MID step type (neither first nor last)
        if self.is_done():
            step_type = 2  # LAST step type
        
        r = self.reward()
        
        return self.TimeStep(
            tf.constant(step_type, shape=(1,), dtype=tf.int32),
            r,
            self.discount,
            tf.convert_to_tensor(self.actions[:, :self.index, :])
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
        r = np.abs(-1.0 * np.log10(1 - fidelity + 1e-100))
        reward = r - self.previous_reward
        self.previous_reward = r
        return tf.constant(reward, shape=(1,), dtype=tf.float32)
    
    def is_done(self):
        """Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        """
        return self.t >= self.T
