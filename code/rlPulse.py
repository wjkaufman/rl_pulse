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

def getPhiFromAction(a):
    return a[..., 0] * 2*np.pi

def getRotFromAction(a):
    return a[..., 1] * 2*np.pi

def getTimeFromAction(a):
    return 10.0**(a[..., 2]*2 - 7) -1e-7

def formatAction(a):
    if len(np.shape(a)) == 1:
        # a single action
        if a[2] != 0:
            return "phi={}pi, rot={}pi, t={}micros".format(\
                    round(getPhiFromAction(a)/np.pi, 1), \
                    round(getRotFromAction(a)/np.pi, 1), \
                    round(getTimeFromAction(a)*1e6, 2))
    elif len(np.shape(a)) == 2:
        str = ""
        for i in range(np.size(a,0)):
            if a[i,2] != 0:
                str += "{}: phi={}pi, rot={}pi, t={}micros\n".format(i, \
                    round(getPhiFromAction(a[i,:])/np.pi, 1), \
                    round(getRotFromAction(a[i,:])/np.pi, 1), \
                    round(getTimeFromAction(a[i,:])*1e6, 2))
        return str
    elif len(np.shape(a)) == 3:
        str = ""
        for i in range(np.size(a,0)):
            str += "===== {} =====\n".format(i)
            for j in range(np.size(a,1)):
                if a[i,j,2] != 0:
                    str += "{}: phi={}pi, rot={}pi, t={}micros\n".format(j, \
                        round(getPhiFromAction(a[i,j,:])/np.pi, 1), \
                        round(getRotFromAction(a[i,j,:])/np.pi, 1), \
                        round(getTimeFromAction(a[i,j,:])*1e6, 2))
        return str
    else:
        print("There was a problem...")
        raise

def printAction(a):
    print(formatAction(a))

def clipAction(a):
    '''Clip the action to give physically meaningful information
    An action a = [phi/2pi, rot/2pi, f(t)], each element in [0,1].
    TODO justify these boundaries, especially for pulse time...
    '''
    return np.array([np.clip(a[0], 0, 1), np.clip(a[1], 0, 1), \
                     np.clip(a[2], np.log10(2), 1)])

def actionNoise(p):
    '''Add noise to actions. Generates a 1x3 array with random values
    
    Arguments:
        p: Parameter to control amount of noise
    '''
#     return np.array([1.0/4*np.random.choice([0,1,-1],p=[1-p,p/2,p/2]), \
#                      1.0/4*np.random.choice([0,1,-1],p=[1-p,p/2,p/2]), \
#                      np.random.normal(0,.25)])
#     return np.array([np.random.normal(0, p/2), \
#                      np.random.normal(0, p/2), \
#                      np.random.normal(0, p/2)])
    return np.array([np.random.uniform(-p/2, p/2), \
                     np.random.uniform(-p/2, p/2), \
                     np.random.uniform(-p/2, p/2)])

def getPropagatorFromAction(N, dim, a, H, X, Y):
    '''Convert an action a into the RF Hamiltonian H.
    
    TODO: change the action encoding to (phi, strength, t) to more easily
        constrain relevant parameters (minimum time, maximum strength)
    
    Arguments:
        a: Action performed on the system. The action is a 1x3 array
            [phi/2pi, rot/2pi, f(t)], where phi specifies the axis of rotation,
            rot specifies the rotation angle (in radians), and t specifies
            the time to perform rotation. f(t) is a function that scales
            relevant time values into the interval [0,1] (or thereabouts).
        H: Time-independent Hamiltonian.
    
    Returns:
        The propagator U corresponding to the time-independent Hamiltonian and
        the RF pulse
    '''
    if a.ndim == 1:
        rotDir = 1
        if a[0] == 0 or a[0] == 1:
            J = X
        elif a[0] == .25:
            J = Y
        elif a[0] == .5:
            J = X
            rotDir = -1
        elif a[0] == .75:
            J = Y
            rotDir = -1
        else:
            # get the angular momentum operator corresponding to rotation axis
            J = ss.getAngMom(np.pi/2, getPhiFromAction(a), N, dim)
        rot = getRotFromAction(a * rotDir)
        time = getTimeFromAction(a)
        return spla.expm(-1j*(H*time + J*rot))
    elif a.ndim == 2:
        # sequence of actions, find composite propagator
        U = np.eye(dim, dtype="complex64")
        for i in range(a.shape[0]):
            if a[i,2] > 0:
                U = getPropagatorFromAction(N, dim, a[i,:], H, X, Y) @ U
        return U
    else:
        print("something went wrong...")
        raise



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
    
    def __init__(self, sDim, aDim, learningRate):
        '''Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space
            aDim: Dimension of action space
            aBounds: Bounds on action space. Should be an aDim*2 object
                specifying min and max values for each dimension
        '''
        self.sDim = sDim
        self.aDim = aDim
        
        self.createNetwork()
        self.optimizer = keras.optimizers.Adam(learningRate)
    
    def createNetwork(self):
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(64, input_shape = (None, self.sDim,), \
            # bias_initializer=tf.random_normal_initializer(stddev=1), \
            # unit_forget_bias=True), \
            ))
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(self.aDim, activation="sigmoid"))
    
    def predict(self, states, training=False):
        '''
        Predict policy values from given states
        
        Arguments:
            states: A batchSize*timesteps*sDim array.
        '''
        if len(np.shape(states)) == 3:
            # predicting on a batch of states
            return self.model(states, training=training)
        elif len(np.shape(states)) == 2:
            # predicting on a single state
            return self.model(np.expand_dims(states,0), training=training)[0]
    
    #@tf.function
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
                    critic.predict(batch[0], \
                                   self.predict(batch[0], training=True), \
                                   training=True))
            # scale gradient by batch size and negate to do gradient ascent
            Qsum = tf.multiply(Qsum, -1.0 / len(batch[0]))
        gradients = g.gradient(Qsum, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, a, polyak=0):
        '''Update the network parameters from another actor, using
        polyak averaging, so that
        theta_self = (1-polyak) * theta_self + polyak * theta_a
        
        Arguments:
            polyak: Polyak averaging parameter between 0 and 1
        '''
        params = self.getParams()
        aParams = a.getParams()
        copyParams = [params[i] * (1-polyak) + aParams[i] * polyak \
                           for i in range(len(params))]
        self.setParams(copyParams)
    
    def calculateDiff(self, a):
        '''Calculate the Frobenius norm for network parameters between network
        and another network.
        '''
        diff = [np.linalg.norm(_[0] - _[1]) for _ in \
                        zip(self.getParams(), a.getParams())]
        diff = np.linalg.norm(diff)
        return diff
        
    

class Critic(object):
    '''Define a Critic that learns the Q-function
    Q: state space * action space -> rewards
    which gives the total maximum expected rewards by choosing the
    state-action pair
    '''
    
    def __init__(self, sDim, aDim, aBounds, gamma, learningRate):
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
        self.optimizer = keras.optimizers.Adam(learningRate)
        self.loss = keras.losses.MeanSquaredError()
    
    def createNetwork(self):
        stateInput = layers.Input(shape=(None, self.sDim,), name="stateInput")
        actionInput = layers.Input(shape=(self.aDim,), name="actionInput")
        stateLSTM = layers.LSTM(64, \
            # bias_initializer=tf.random_normal_initializer(stddev=1), \
            # unit_forget_bias=True, \
            )(stateInput)
        x = layers.concatenate([stateLSTM, actionInput])
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(1, activation="relu", name="output")(x)
        self.model = keras.Model(inputs=[stateInput, actionInput], \
                            outputs=[output])
    
    def predict(self, states, actions, training=False):
        '''
        Predict Q-values for given state-action inputs
        '''
        if len(np.shape(states)) == 3:
            # predicting on a batch of states/actions
            return self.model({"stateInput": states,"actionInput": actions}, \
                              training=training)
        elif len(np.shape(states)) == 2:
            # predicting on a single state/action
            return self.model({"stateInput": np.expand_dims(states,0), \
                               "actionInput": np.expand_dims(actions,0)}, \
                              training=training)[0]
    
    #@tf.function
    def trainStep(self, batch, actorTarget, criticTarget):
        '''Trains the critic's Q-network one step
        using the gradient specified by the DDPG algorithm
        
        Arguments:
            batch: A batch of experiences from the replayBuffer
            actorTarget: Target actor
            criticTarget: Target critic
        '''
        targets = batch[2] + self.gamma * (1-batch[4]) * \
            criticTarget.predict(batch[3], actorTarget.predict(batch[3]), \
                                 training=True)
        # calculate gradient according to DDPG algorithm
        with tf.GradientTape() as g:
            predictions = self.predict(batch[0], batch[1], training=True)
            predLoss = self.loss(predictions, targets)
            predLoss = tf.math.multiply(predLoss, 1.0 / len(batch[0]))
        gradients = g.gradient(predLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, a, polyak=0):
        '''Update the network parameters from another network, using
        polyak averaging, so that
        theta_self = (1-polyak) * theta_self + polyak * theta_a
        
        Arguments:
            polyak: Polyak averaging parameter between 0 and 1
        '''
        params = self.getParams()
        aParams = a.getParams()
        copyParams = [params[i] * (1-polyak) + aParams[i] * polyak \
                           for i in range(len(params))]
        self.setParams(copyParams)
    
    def calculateDiff(self, c):
        '''Calculate the Frobenius norm for network parameters between network
        and another network.
        '''
        diff = [np.linalg.norm(_[0] - _[1]) for _ in \
                        zip(self.getParams(), c.getParams())]
        diff = np.linalg.norm(diff)
        return diff

    

class Environment(object):
    
    def __init__(self, N, dim, sDim, Htarget, X, Y):
        self.N = N
        self.dim = dim
        self.sDim = sDim
        self.Htarget = Htarget
        self.X = X
        self.Y = Y
        
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
        self.state = np.zeros((16, self.sDim), dtype="float32")
    
    def getState(self):
        return np.copy(self.state)
    
    def evolve(self, a, Hint):
        '''Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        '''
        phi = getPhiFromAction(a)
        rot = getRotFromAction(a)
        dt  = getTimeFromAction(a)
        if dt > 0:
            self.Uexp = getPropagatorFromAction(self.N, self.dim, a, \
                            Hint, self.X, self.Y) @ self.Uexp
            self.Utarget = ss.getPropagator(self.Htarget, getTimeFromAction(a)) @ \
                            self.Utarget
            self.t += getTimeFromAction(a)
            self.state[np.where(self.state[:,2] == 0)[0][0],:] = a
    
    def reward(self):
        return -1.0 * np.log10(1 + 1e-12 - \
                    np.minimum(ss.fidelity(self.Utarget, self.Uexp), 1))
    
    def reward1(self, beta):
        return -1.0 * np.log10(1 + 1e-12 - np.minimum(1, \
                np.exp(beta * self.t) * \
                np.minimum(ss.fidelity(self.Utarget, self.Uexp), 1)))
    
    def reward2(self):
        return -1.0 * np.log10(1 + 1e-12 - np.minimum( \
            np.power(ss.fidelity(self.Utarget, self.Uexp), 2e-5/self.t), 1))
    
    def isDone(self):
        '''Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        TODO modify this when I move on from constrained (4-pulse) sequences
        '''
        return (self.t >= 4*.25e-6 + 6*3e-6) or \
               (np.sum(self.state[:,2] == 0) == 0)
