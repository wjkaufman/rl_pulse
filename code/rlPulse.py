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
            d: TODO figure this out lol
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
            TODO what is d???
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
    
    def createNetwork(self):
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(64, input_shape = (self.sDim,)))
        self.model.add(layers.Dense(64))
        self.model.add(layers.Dense(self.aDim))
        
        # TODO add scaling to output to put it within aBounds
        
        self.optimizer = keras.optimizers.Adam(0.001)
    
    @tf.function
    def trainStep(self, batch, critic):
        '''Trains the actor's policy network one step
        using the gradient specified by the DDPG algorithm
        
        Arguments:
            batch: A batch of experiences from the replayBuffer
            critic: A critic to estimate the Q-function
        '''
        states = batch[0]
        
        # calculate gradient according to DDPG algorithm
        with tf.GradientTape() as g:
            Qsum = tf.math.reduce_sum( \
                    critic.predict(states, self.predict(states)))
            # scale gradient by batch size and negate to do gradient ascent
            Qsum = tf.multiply(Qsum, -1.0 / len(batch[0]))
        gradients = g.gradient(Qsum, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, a, polyak=1):
        '''Copy the network parameters from another actor, using
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
    
    def createNetwork(self):
        # TODO write this out...!!!!
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(64, input_shape = (self.sDim,)))
        self.model.add(layers.Dense(64))
        self.model.add(layers.Dense(self.aDim))
        
        # TODO add scaling to output to put it within aBounds

        self.optimizer = keras.optimizers.Adam(0.001)
        self.loss = keras.losses.MeanSquaredError()
    
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
                criticTarget.predict(batch[3], actorTarget.predict(batch[3]))
            predLoss = self.loss(self.predict(batch[0], batch[1]), targets)
            predLoss = tf.math.multiply(predLoss, 1.0 / len(batch[0]))
        gradients = g.gradient(predLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, a, polyak=1):
        '''Copy the network parameters from another actor, using
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
        
