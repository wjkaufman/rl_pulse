# implements a very simple n-arm bandit example to make sure the RL
# code is running properly

import numpy as np

class BanditEnv(object):
    def __init__(self, mean, sd, sDim):
        """Create bandits with normal reward distributions according to mean, sd
        """
        
        self.mean = mean
        self.sd = sd
        self.sDim = sDim
        self.reset()
    
    def getState(self):
        return self.state
    
    def act(self, action):
        # print(f'action: {action.action}')
        n = np.nonzero(action.action)[0][0]
        # print(f'n: {n}')
        self.r = np.random.normal(loc=self.mean[n], scale=self.sd[n])
        # print(f'r: {self.r}')
        self.state[self.ind] = action.action
        self.ind += 1
    
    def reward(self):
        return self.r
    
    def reset(self):
        self.state = np.zeros((16, self.sDim), dtype='float32')
        self.r = 0
        self.ind = 0
    
    def isDone(self):
        return self.ind > 15
