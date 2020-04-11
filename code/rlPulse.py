'''
Actor and critic classes adapted from
https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

'''
import numpy as np
import scipy.linalg as spla
import random
from collections import deque
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import spinSimulation as ss



def actionToPropagator(N, dim, a, H):
    '''Convert an action a into the RF Hamiltonian H.
    
    TODO: change the action encoding to (phi, strength, t) to more easily
        constrain relevant parameters (minimum time, maximum strength)
    
    Arguments:
        a: Action performed on the system. The action is a 1x3 array
            [phi, rot, t], where phi specifies the axis of rotation, rot
            specifies the rotation angle (in radians), and t specifies the time
            to perform rotation.
        H: Time-independent Hamiltonian.
    
    Returns:
        The propagator U corresponding to the time-independent Hamiltonian and
        the RF pulse
    '''
    if a[0] == 0:
        J = X
    elif a[0] == np.pi/2:
        J = Y
    elif a[0] == np.pi:
        J = X
        a[1] *= -1.0
    elif a[0] == 3*np.pi/2:
        J = Y
        a[1] *= -1.0
    else:
        # get the angular momentum operator J corresponding to the axis of rotation
        J = ss.getAngMom(np.pi/2, a[0], N, dim)
    # and keep going with implementation...
    return spla.expm(-1j*2*np.pi*(H*a[2] + J*a[1]))

def clipAction(a):
    '''Clip the action to give physically meaningful information
    An action a = [phi, rot, time], phi in [0,2*pi], rot in [0,2pi], time > 0.
    '''
    return np.array([np.mod(a[0], 2*pi), np.mod(a[1], 2*pi), np.max(a[3], 0)])


class ReplayBuffer(object):
    '''Define a ReplayBuffer object to store experiences for training
    '''
    
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.count = 0
        self.buffer = deque()
        
    def add(self, s, a, r, s1, d):
        '''Add an experience to the buffer.
        If the buffer is full, pop an old experience and
        add the new experience.
        
        Arguments:
            s: Old environment state on which action was performed
            a: Action performed
            r: Reward from state-action pair
            s1: New environment state from state-action pair
            d: Is s1 terminal (if so, end the episode)
        '''
        exp = (s,a,r,s1,d)
        if self.count < self.bufferSize:
            self.buffer.append(exp)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)
    
    def size(self):
        return self.count
    
    def getSampleBatch(self, batchSize):
        '''Get a sample batch from the replayBuffer
        
        Arguments:
            batchSize: Size of the sample batch to return. If the replay buffer
                doesn't have batchSize elements, return the entire buffer
        
        Returns:
            A tuple of arrays (states, actions, rewards, new states, and d)
        '''
        batch = []
        
        if self.count < batchSize:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batchSize)
        
        sBatch = np.array([_[0] for _ in batch])
        aBatch = np.array([_[1] for _ in batch])
        rBatch = np.array([_[2] for _ in batch])
        s1Batch = np.array([_[3] for _ in batch])
        dBatch = np.array([_[4] for _ in batch])
        
        return sBatch, aBatch, rBatch, s1Batch, dBatch
    
    def clear(self):
        self.buffer.clear()
        self.count = 0



class Actor(object):
    '''Define an Actor object that learns the deterministic policy function
    pi(s): state space -> action space
    '''
    
    def __init__(self, sDim, aDim, aBounds):
        '''Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space
            aDim: Dimension of action space
            aBounds: Bounds on action space. Should be an aDim*2 object
                specifying min and max values for each dimension
        '''
        self.sDim = sDim
        self.aDim = aDim
        self.aBounds = aBounds
        
        self.createNetwork()
        self.optimizer = keras.optimizers.Adam(0.001)
    
    def createNetwork(self):
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(64, input_shape = (None, self.sDim,)))
        self.model.add(layers.Dense(64))
        self.model.add(layers.Dense(self.aDim))
        
        # TODO add scaling to output to put it within aBounds
    
    def predict(self, states):
        '''
        Predict policy values from given states
        '''
        return self.model.predict(states)
    
    @tf.function
    def trainStep(self, batch, critic):
        '''Trains the actor's policy network one step
        using the gradient specified by the DDPG algorithm
        
        Arguments:
            batch: A batch of experiences from the replayBuffer. `batch` is
                a tuple: (state, action, reward, new state, is terminal?).
            critic: A critic to estimate the Q-function
        '''
        # calculate gradient according to DDPG algorithm
        with tf.GradientTape() as g:
            Qsum = tf.math.reduce_sum( \
                    critic.predict({"stateInput": batch[0], \
                                "actionInput": self.predict(batch[0])}))
            # scale gradient by batch size and negate to do gradient ascent
            Qsum = tf.multiply(Qsum, -1.0 / len(batch[0]))
        gradients = g.gradient(Qsum, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def updateParams(self, a, polyak=1):
        '''Update the network parameters from another actor, using
        polyak averaging, so that
        theta_self = polyak * theta_self + (1-polyak) * theta_a
        
        Arguments:
            polyak: Polyak averaging parameter between 0 and 1
        '''
        params = self.getParams()
        aParams = a.getParams()
        updateParams = [params[i] * polyak + aParams[i] * (1-polyak) \
                           for i in range(len(params))]
        self.setParams(updateParams)
        
    

class Critic(object):
    '''Define a Critic that learns the Q-function
    Q: state space * action space -> rewards
    which gives the total maximum expected rewards by choosing the
    state-action pair
    '''
    
    def __init__(self, sDim, aDim, aBounds, gamma):
        '''Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space
            aDim: Dimension of action space
            aBounds: Bounds on action space. Should be an aDim*2 object
                specifying min and max values for each dimension
            gamma: discount rate for future rewards
        '''
        self.sDim = sDim
        self.aDim = aDim
        self.aBounds = aBounds
        self.gamma = gamma
        
        self.createNetwork()
        self.optimizer = keras.optimizers.Adam(0.001)
        self.loss = keras.losses.MeanSquaredError()
    
    def createNetwork(self):
        stateInput = layers.Input(shape=(None, self.sDim,), name="stateInput")
        actionInput = layers.Input(shape=(self.aDim,), name="actionInput")
        stateLSTM = layers.LSTM(64)(stateInput)
        x = layers.concatenate([stateLSTM, actionInput])
        x = layers.Dense(64)(x)
        output = layers.Dense(1, name="output")(x)
        self.model = keras.Model(inputs=[stateInput, actionInput], \
                            outputs=[output])
    
    def predict(self, states, actions):
        '''
        Predict Q-values for given state-action inputs
        '''
        return self.model.predict({"stateInput": states,"actionInput": actions})
    
    @tf.function
    def trainStep(self, batch, actorTarget, criticTarget):
        '''Trains the critic's Q-network one step
        using the gradient specified by the DDPG algorithm
        
        Arguments:
            batch: A batch of experiences from the replayBuffer
            actorTarget: Target actor
            criticTarget: Target critic
        '''
        
        # calculate gradient according to DDPG algorithm
        with tf.GradientTape() as g:
            targets = batch[2] + self.gamma * (1-batch[4]) * \
                criticTarget.predict({"stateInput": batch[3], \
                            "actionInput": actorTarget.predict(batch[3])})
            predLoss = self.loss(self.predict({"stateInput": batch[0], \
                                    "actionInput": batch[1]}), targets)
            predLoss = tf.math.multiply(predLoss, 1.0 / len(batch[0]))
        gradients = g.gradient(predLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def updateParams(self, a, polyak=1):
        '''Update the network parameters from another actor, using
        polyak averaging, so that
        theta_self = polyak * theta_self + (1-polyak) * theta_a
        
        Arguments:
            polyak: Polyak averaging parameter between 0 and 1
        '''
        params = self.getParams()
        aParams = a.getParams()
        updateParams = [params[i] * polyak + aParams[i] * (1-polyak) \
                            for i in range(len(params))]
        self.setParams(updateParams)

    

class Environment(object):
    
    def __init__(self, N, dim, sDim, Htarget):
        self.N = N
        self.dim = dim
        self.sDim = sDim
        self.Htarget = Htarget
        
        self.reset()
    
    def reset(self):
        '''Resets the environment by setting all propagators to the identity
        and setting t=0
        '''
        # initialize propagators to identity
        self.Uexp = np.eye(self.dim, dtype="complex64")
        self.Utarget = np.copy(self.Uexp)
        # initialize time t=0
        self.t = 0
        
        # for network training, define the "state" (sequence of actions)
        self.state = np.zeros((0, self.sDim))
    
    def getState(self):
        return np.copy(self.state)
    
    def evolve(self, a, Hint):
        '''Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        '''
        self.Uexp = actionToPropagator(self.N, self.dim, a, Hint) @ self.Uexp
        self.Utarget = ss.getPropagator(self.Htarget, a[2]) @ self.Utarget
        self.state = np.append(self.state, a)
    
    def reward(self):
        return -1.0 * np.log10(1 + 1e-9 - ss.fidelity(self.Utarget, self.Uexp))
        
    def isDone(self):
        '''Returns true if the environment has reached a certain time point
        TODO modify this when I move on from constrained (4-pulse) sequences
        '''
        return np.sum(self.state, 0)[2] >= 4*.25e-6 + 6*3e-6
