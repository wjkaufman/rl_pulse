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
    return a[..., 0] * np.pi/2

def getRotFromAction(a):
    return a[..., 1] * 2*np.pi

def getTimeFromAction(a):
    '''Get the time (in seconds) from the action encoding.
    
    Ideally want action-time mappings to be 0 -> 0, 1 -> 5e-6.
    '''
    return 10.0**((a[..., 2]+1)*0.853785 - 7) -1e-7

def formatAction(a):
    if len(np.shape(a)) == 1:
        # a single action
        if a[1] != 0:
            return f"phi={getPhiFromAction(a)/np.pi:.02f}pi, " + \
                f" rot={getRotFromAction(a)/np.pi:.02f}pi, " + \
                f"t={getTimeFromAction(a)*1e6:.02f} microsec"
        else:
            if a[2] != 0:
                return f'delay, t={getTimeFromAction(a)*1e6:.02f} microsec'
            else:
                return ''
    elif len(np.shape(a)) == 2:
        str = ""
        for i in range(np.size(a,0)):
            strA = formatAction(a[i,:])
            if strA != '':
                str += f'{i}: ' + strA + '\n'
        return str
    elif len(np.shape(a)) == 3:
        str = ""
        for i in range(np.size(a,0)):
            str += f"===== {i} =====\n" + formatAction(a[i,:,:]) + '\n'
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
    return np.array([np.mod(a[0], 1), np.clip(a[1], 0, 1), \
                     np.clip(a[2], 0, 1)])

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
        rot = getRotFromAction(a) * rotDir
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



class NoiseProcess(object):
    '''A noise process that can have temporal autocorrelation
    
    Scale should be a number between 0 and 1.
    
    TODO need to add more sophisticated noise here...
    '''
    
    def __init__(self, scale):
        self.scale = scale
    
    def copy(self):
        return NoiseProcess(self.scale)
    
    def getNoise(self):
        return np.array( \
            [np.random.normal(loc=0, scale=.1*self.scale) + \
                np.random.choice([-.25,.25,.5,0], \
                p=[self.scale/3,self.scale/3,self.scale/3,\
                            1-self.scale]), \
             np.random.normal(loc=0, scale=.1*self.scale) + \
                np.random.choice([-.5,-.25,.25,.5,0], \
                p=[self.scale/4,self.scale/4,\
                            self.scale/4,self.scale/4,1-self.scale]), \
             np.random.normal(loc=0, scale=.1*self.scale) + \
                np.random.choice([-.5,.5,0], \
                p=[self.scale/2,self.scale/2,1-self.scale])])

class ReplayBuffer(object):
    '''Define a ReplayBuffer object to store experiences for training
    '''
    
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.size = 0
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
        if self.size < self.bufferSize:
            self.buffer.append(exp)
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)
    
    def getSampleBatch(self, batchSize):
        '''Get a sample batch from the replayBuffer
        
        Arguments:
            batchSize: Size of the sample batch to return. If the replay buffer
                doesn't have batchSize elements, return the entire buffer
        
        Returns:
            A tuple of arrays (states, actions, rewards, new states, and d)
        '''
        batch = []
        
        if self.size < batchSize:
            batch = random.sample(self.buffer, self.size)
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
        self.size = 0

def mutateMat(mat, mutateStrength=1, mutateFrac=.1, \
        superMutateProb=.01, resetProb=.01):
    '''Method to perform mutations on a given nd-array
    
    Arguments:
        mat: Matrix on which to perform mutations.
        mutateStrength: Strength of mutation. Corresponds to the variance of
            the random number that the weight is multiplied by.
        superMutateProb: Probability of a "super-mutation." This is the
            probability _given_ that a mutation occurs.
        resetProb: Probability of resetting the weight. This is the
            probability _given_ that a mutation occurs.
    '''
    # choose which elements to mutate
    mutateInd = np.random.choice(mat.size, \
            int(mat.size * mutateFrac), replace=False)
    superMutateInd = mutateInd[0:int(mat.size*mutateFrac*superMutateProb)]
    resetInd = mutateInd[int(mat.size*mutateFrac*superMutateProb):\
        int(mat.size*mutateFrac*(superMutateProb + resetProb))]
    mutateInd = mutateInd[int(mat.size*mutateFrac*(superMutateProb+resetProb)):]
    
    # perform mutations on mat
    mat[np.unravel_index(superMutateInd, mat.shape)] *= \
        np.random.normal(scale=100*mutateStrength, size=superMutateInd.size)
    mat[np.unravel_index(resetInd, mat.shape)] = \
        np.random.normal(size=resetInd.size)
    mat[np.unravel_index(mutateInd, mat.shape)] *= \
        np.random.normal(scale=mutateStrength, size=mutateInd.size)



class Actor(object):
    '''Define an Actor object that learns the deterministic policy function
    pi(s): state space -> action space
    '''
    
    def __init__(self, sDim, aDim, learningRate):
        '''Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space.
            aDim: Dimension of action space.
            learningRate: Learning rate for optimizer.
        '''
        self.sDim = sDim
        self.aDim = aDim
        self.learningRate = learningRate
        self.model = None
        
        self.optimizer = keras.optimizers.Adam(learningRate)
    
    def createNetwork(self, lstmLayers, fcLayers, lstmUnits, fcUnits):
        '''Create the network
        
        Arguments:
            lstmLayers: The number of LSTM layers to process state input
            fcLayers: The number of fully connected layers
        '''
        self.model = keras.Sequential()
        # add LSTM layers
        if lstmLayers == 1:
            self.model.add(layers.LSTM(lstmUnits,\
                input_shape=(None,self.sDim,), \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                ))
        elif lstmLayers == 2:
            self.model.add(layers.LSTM(lstmUnits, \
                input_shape=(None,self.sDim,), \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True, \
                ))
            self.model.add(layers.LSTM(lstmUnits))
        elif lstmLayers > 2:
            self.model.add(layers.LSTM(lstmUnits, \
                input_shape=(None, self.sDim,), \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True, \
                ))
            for i in range(lstmLayers-2):
                self.model.add(layers.LSTM(lstmUnits, \
                    # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                    # unit_forget_bias=True, \
                    return_sequences=True))
            self.model.add(layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True,\
                ))
        else:
            print("Problem making the network...")
            raise
        # add batch normalization layer
        self.model.add(layers.BatchNormalization())
        # add fully connected layers
        for i in range(fcLayers):
            # self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dense(fcUnits, activation="relu"))
        self.model.add(layers.Dense(self.aDim, activation="tanh", \
            bias_initializer=tf.random_normal_initializer(stddev=0.1)))
    
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
    
    def save_weights(self, filepath):
        '''Save model weights in ckpt format
        '''
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def getParams(self):
        return self.model.get_weights()
    
    def setParams(self, params):
        return self.model.set_weights(params)
    
    def copyParams(self, actor, polyak=0):
        '''Update the network parameters from another actor, using
        polyak averaging, so that
        theta_self = (1-polyak) * theta_self + polyak * theta_a
        
        Arguments:
            
            polyak: Polyak averaging parameter between 0 and 1
        '''
        params = self.getParams()
        aParams = actor.getParams()
        copyParams = [params[i] * (1-polyak) + aParams[i] * polyak \
                           for i in range(len(params))]
        self.setParams(copyParams)
    
    def copy(self):
        '''Copy the actor and return a new actor with same model
        and model parameters.
        '''
        copy = Actor(self.sDim, self.aDim, self.learningRate)
        copy.model = keras.models.clone_model(self.model)
        copy.setParams(self.getParams())
        return copy
    
    def paramDiff(self, a):
        '''Calculate the Frobenius norm for network parameters between network
        and another network.
        '''
        diff = [np.mean((_[0] - _[1])**2) for _ in \
                        zip(self.getParams(), a.getParams())]
        # diff = np.linalg.norm(diff)
        return diff
    
    def evaluate(self, env, replayBuffer, noiseProcess, numEval=1):
        '''Perform a complete play-through of an episode, and
        return the total rewards from the episode.
        '''
        f = 0.
        for i in range(numEval):
            env.reset()
            env.evolve(np.array([0,0,1])) # start with delay
            s = env.getState()
            done = False
            while not done:
                a = self.predict(s)
                if noiseProcess is not None:
                    a += noiseProcess.getNoise()
                a = clipAction(a)
                env.evolve(a)
                env.evolve(np.array([0,0,1])) # add delay
                r = env.reward()
                s1 = env.getState()
                done = env.isDone()
                replayBuffer.add(s,a,r,s1, done)
                s = s1
                f = np.maximum(f, r)
        return f
    
    def test(self, env):
        '''Test the actor's ability without noise. Return the actions it
        performs and the rewards it gets through the episode
        '''
        rMat = []
        env.reset()
        env.evolve(np.array([0,0,1])) # add delay
        s = env.getState()
        done = False
        while not done:
            a = clipAction(self.predict(s))
            env.evolve(a)
            env.evolve(np.array([0,0,1])) # add delay
            rMat.append(env.reward())
            s = env.getState()
            done = env.isDone()
        return s, rMat
    
    def crossover(self, p1, p2, weight=0.5):
        '''Perform evolutionary crossover with two parent actors. Using
        both parents' parameters, copies their "genes" to this actor.
        
        Many choices for crossover methods exist. This implements the simplest
        uniform crossover, which picks "genes" from either parent with
        probabilities weighted by
        
        Arguments:
            p1, p2: Actors whose parameters are crossed, then copied to self.
            weight: Probability of selecting p1's genes to pass to child.
                Should probably be dependent on the relative fitness of the
                two parents.
        '''
        childParams = self.getParams()
        p1Params = p1.getParams()
        p2Params = p2.getParams()
        for i in range(len(p1Params)):
            if np.random.rand() < weight:
                childParams[i] = p1Params[i]
            else:
                childParams[i] = p2Params[i]
        self.setParams(childParams)
    
    def mutate(self, mutateStrength=1, mutateFrac=.1, \
    superMutateProb=.01, resetProb=.01):
        '''Mutate the parameters for the neural network.
        
        Arguments:
            mutateFrac: Fraction of weights that will be mutated.
            superMutateProb: Probability that a "super mutation" occurs
                (i.e. the weight is multiplied by a higher-variance random
                number).
            resetProb: Probability that the weight is reset to a random value.
        '''
        params = self.getParams()
        for i in range(len(params)):
            mutateMat(params[i], mutateStrength, mutateFrac, superMutateProb, \
                resetProb)
    


class Critic(object):
    '''Define a Critic that learns the Q-function
    Q: state space * action space -> rewards
    which gives the total maximum expected rewards by choosing the
    state-action pair
    '''
    
    def __init__(self, sDim, aDim, gamma, learningRate):
        '''Initialize a new Actor object
        
        Arguments:
            sDim: Dimension of state space
            aDim: Dimension of action space
            gamma: discount rate for future rewards
        '''
        self.sDim = sDim
        self.aDim = aDim
        self.gamma = gamma
        self.learningRate = learningRate
        
        self.model = None
        self.optimizer = keras.optimizers.Adam(learningRate)
        self.loss = keras.losses.MeanSquaredError()
    
    def createNetwork(self, lstmLayers, fcLayers, lstmUnits, fcUnits):
        '''Create the network
        
        Arguments:
            lstmLayers: The number of LSTM layers to process state input
            fcLayers: The number of fully connected layers
        '''
        stateInput = layers.Input(shape=(None, self.sDim,), name="stateInput")
        actionInput = layers.Input(shape=(self.aDim,), name="actionInput")
        # add LSTM layers
        if lstmLayers == 1:
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True,
                )(stateInput)
        elif lstmLayers == 2:
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True)(stateInput)
            stateLSTM = layers.LSTM(lstmUnits)(stateLSTM)
        elif lstmLayers > 2:
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                return_sequences=True)(stateInput)
            for i in range(lstmLayers-2):
                stateLSTM=layers.LSTM(lstmUnits, \
                    # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                    # unit_forget_bias=True, \
                    return_sequences=True)(stateLSTM)
            stateLSTM = layers.LSTM(lstmUnits, \
                # bias_initializer=tf.random_normal_initializer(stddev=.05), \
                # unit_forget_bias=True, \
                )(stateLSTM)
        else:
            print("Problem making the network...")
            raise
        x = layers.BatchNormalization()(stateLSTM)
        # concatenate state, action inputs
        x = layers.concatenate([stateLSTM, actionInput])
        # add fully connected layers
        for i in range(fcLayers):
            # x = layers.BatchNormalization()(x)
            x = layers.Dense(fcUnits, activation="relu")(x)
        output = layers.Dense(1, name="output")(x)
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
            criticTarget.predict(batch[3], actorTarget.predict(batch[3]))
        # calculate gradient according to DDPG algorithm
        with tf.GradientTape() as g:
            predictions = self.predict(batch[0], batch[1], training=True)
            predLoss = self.loss(predictions, targets)
            predLoss = tf.math.multiply(predLoss, 1.0 / len(batch[0]))
        gradients = g.gradient(predLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients( \
                zip(gradients, self.model.trainable_variables))
    
    def save_weights(self, filepath):
        '''Save model weights in ckpt format
        '''
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
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
    
    def copy(self):
        '''Copy the critic and return a new critic with same model
        and model parameters.
        '''
        copy = Critic(self.sDim, self.aDim, self.gamma, self.learningRate)
        copy.model = keras.models.clone_model(self.model)
        copy.setParams(self.getParams())
        return copy
    
    def paramDiff(self, c):
        '''Calculate the Frobenius norm for network parameters between network
        and another network.
        '''
        diff = [np.mean((_[0] - _[1])**2) for _ in \
                        zip(self.getParams(), c.getParams())]
        # diff = np.linalg.norm(diff)
        return diff

    

class Population(object):
    '''Population of actors, for evolutionary reinforcement learning.
    
    '''
    
    def __init__(self, size=10):
        self.size = size
        self.fitnesses = np.full((self.size,), -1e100, dtype=float)
        self.pop = np.full((self.size,), None, dtype=object)
    
    def startPopulation(self, sDim, aDim, learningRate, lstmLayers, fcLayers, \
        lstmUnits, fcUnits):
        for i in range(self.size):
            self.pop[i] = Actor(sDim, aDim, learningRate)
            self.pop[i].createNetwork(lstmLayers,fcLayers,lstmUnits,fcUnits)
    
    def evaluate(self, env, replayBuffer, noiseProcess, numEval=1):
        '''Evaluate the fitnesses of each member of the population.
        
        '''
        for i in range(self.size):
            print(f'evaluating individual {i}/{self.size},\t', end='')
            self.fitnesses[i] = self.pop[i].evaluate(env, replayBuffer, \
                                        noiseProcess, numEval)
            print(f'fitness is {self.fitnesses[i]:.02f}')
        # with mp.Pool() as pool:
        #     print('getting results')
        #     results = pool.map(f, self.pop)
        #     print(f'got results {results}')
    
    def iterate(self, eliteFrac=0.1, tourneyFrac=.2, crossoverProb=.25, \
        mutateProb = .25, mutateStrength=1, mutateFrac=.1, \
        superMutateProb=.01, resetProb=.01):
        '''Iterate the population to create the next generation
        of members.
        
        Arguments:
            eliteFrac: Fraction of total population that will be marked as
                "elites".
            tourneyFrac: Fraction of population to include in each tournament
                ("tourney").
        '''
        # sort population by fitness
        indSorted = np.argsort(self.fitnesses)
        numElite = int(np.ceil(self.size * eliteFrac))
        elites = self.pop[indSorted[(-numElite):]]
        eliteFitness = self.fitnesses[indSorted[-numElite:]]
        # perform tournament selection to get rest of population
        selected = np.full((self.size-numElite), None, dtype=object)
        selectedFitness = np.full((self.size-numElite), None, dtype=float)
        tourneySize = int(np.ceil(self.size * tourneyFrac))
        for i in range(self.size-numElite):
            # pick random subset of population for tourney
            ind = np.random.choice(self.size, tourneySize, replace=False)
            # pick the winner according to highest fitness
            indWinner = ind[np.argmax(self.fitnesses[ind])]
            winner = self.pop[indWinner]
            winnerFitness = self.fitnesses[indWinner]
            selectedFitness[i] = winnerFitness
            if winner not in selected:
                selected[i] = winner
            else:
                selected[i] = winner.copy()
        
        # do crossover/mutations with individuals in selected
        for s, sf in zip(selected, selectedFitness):
            if np.random.rand() < crossoverProb:
                eInd = np.random.choice(numElite)
                e = elites[eInd]
                ef = eliteFitness[eInd]
                relativeFitness = np.exp(ef) / (np.exp(ef) + np.exp(sf))
                s.crossover(e, s, weight=relativeFitness)
            if np.random.rand() < mutateProb:
                s.mutate(mutateStrength, mutateFrac, \
                superMutateProb, resetProb)
            
        # then reassign them to the population
        self.pop[:numElite] = elites
        self.pop[numElite:] = selected
    
    def sync(self, newMember):
        '''Replace the weakest (lowest-fitness) member with a new member.
        
        '''
        ind = np.argmin(self.fitnesses)
        self.pop[ind] = newMember.copy()
        self.fitnesses[ind] = -1e100
        


class Environment(object):
    
    def __init__(self, N, dim, coupling, delta, sDim, Htarget, X, Y):
        self.N = N
        self.dim = dim
        self.coupling = coupling
        self.delta = delta
        self.sDim = sDim
        self.Htarget = Htarget
        self.X = X
        self.Y = Y
        
        self.reset()
    
    def reset(self):
        '''Resets the environment by setting all propagators to the identity
        and setting t=0
        '''
        # randomize dipolar couplings and get Hint
        _, self.Hint = ss.getAllH(self.N, self.dim, self.coupling, self.delta)
        # initialize propagators to identity
        self.Uexp = np.eye(self.dim, dtype="complex64")
        self.Utarget = np.copy(self.Uexp)
        # initialize time t=0
        self.t = 0
        
        # for network training, define the "state" (sequence of actions)
        self.state = np.zeros((32, self.sDim), dtype="float32")
        self.tInd = 0 # keep track of time index in state
    
    def copy(self):
        '''Return a copy of the environment
        '''
        return Environment(self.N, self.dim, self.coupling, self.delta, \
            self.sDim, self.Htarget, self.X, self.Y)
        
    
    def getState(self):
        return np.copy(self.state)
    
    def evolve(self, a):
        '''Evolve the environment corresponding to an action and the
        time-independent Hamiltonian
        '''
        dt  = getTimeFromAction(a)
        if self.tInd < np.size(self.state, 0):
            self.Uexp = getPropagatorFromAction(self.N, self.dim, a, \
                            self.Hint, self.X, self.Y) @ self.Uexp
            if dt > 0:
                self.Utarget = ss.getPropagator(self.Htarget, dt) @ \
                                self.Utarget
                self.t += dt
            self.state[self.tInd,:] = a
            self.tInd += 1
        else:
            print('ran out of room in state array, not evolving state')
    
    def reward(self):
        return -1.0 * np.log10((1-ss.fidelity(self.Utarget,self.Uexp))+1e-100)

        # isTimeGood = 1/(1 + np.exp((15e-6-self.t)/2e-6))
        # return -1.0 * isTimeGood * np.log10((1 - \
        #     np.power(ss.fidelity(self.Utarget, self.Uexp), 20e-6/self.t)) + \
        #     1e-100)
        #
        # isTimeGood = self.t >= 15e-6
        # isDelay = 1-self.state[self.tInd-1,1] # if there's no rotation
        # return -1.0 * isTimeGood * isDelay * np.log10((1 - \
        #     np.power(ss.fidelity(self.Utarget, self.Uexp), 20e-6/self.t)) + \
        #     1e-100)
    
    def isDone(self):
        '''Returns true if the environment has reached a certain time point
        or once the number of state variable has been filled
        TODO modify this when I move on from constrained (4-pulse) sequences
        '''
        return (self.t >= 50e-6) or \
            (np.sum((self.state[:,1] == 0)*(self.state[:,2] == 0)) == 0)
