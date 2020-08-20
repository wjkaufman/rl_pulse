import numpy as np
from scipy import linalg
import spin_simulation as ss


class Node(object):
    
    def __init__(self, propagator, sequence='', depth=0):
        self.propagator = propagator
        self.sequence = sequence
        self.depth = depth
        
        self.children = {}
    
    def has_children(self):
        return len(self.children) > 0
    
    def evaluate(self, Utarget, actions, reward_dict, max_depth=6):
        '''If the node isn't at max_depth, then create children and
        evaluate each individually. If the node is at max_depth, then
        calculate the reward and add the sequence/reward pair to
        reward_dict.
        
        Arguments:
            actions: An array of unitary operators representing all actions
                that can be applied to the system.
        Returns: The maximum reward seen by the node or its children, and
            the corresponding sequence.
            
        '''
        if self.depth < max_depth:
            max_reward = 0
            max_reward_sequence = ''
            for i, action in enumerate(actions):
                propagator = action @ self.propagator
                child = Node(propagator,
                             self.sequence + str(i),
                             depth=self.depth + 1)
                r, s = child.evaluate(Utarget, actions, reward_dict, max_depth)
                if r > max_reward:
                    max_reward = r
                    max_reward_sequence = s
            return max_reward, max_reward_sequence
        else:
            fidelity = ss.fidelity(Utarget, self.propagator)
            reward = - np.log10(1.0 - fidelity + 1e-100)
            reward_dict[self.sequence] = reward
            return reward, self.sequence


def evaluate_from_root(
        N=4,
        dim=2**4,
        coupling=1e3,
        delta=500,
        delay=10e-6,
        pulse_width=1e-6
        ):
    (X, Y, Z) = ss.get_total_spin(N, dim)
    H_target = ss.get_H_WHH_0(X, Y, Z, delta)
    _, Hint = ss.get_H(N, dim, coupling, delta)
    
    Utarget = ss.get_propagator(H_target, 6*(pulse_width + delay))
    
    # define actions
    Udelay = linalg.expm(-1j*(Hint*(pulse_width + delay)))
    Ux = linalg.expm(-1j*(X*np.pi/2 + Hint*pulse_width))
    Uxbar = linalg.expm(-1j*(X*-np.pi/2 + Hint*pulse_width))
    Uy = linalg.expm(-1j*(Y*np.pi/2 + Hint*pulse_width))
    Uybar = linalg.expm(-1j*(Y*-np.pi/2 + Hint*pulse_width))
    Ux = Udelay @ Ux
    Uxbar = Udelay @ Uxbar
    Uy = Udelay @ Uy
    Uybar = Udelay @ Uybar
    actions = [Ux, Uxbar, Uy, Uybar, Udelay]
    
    reward_dict = {}
    
    root = Node(np.eye(dim, dtype='complex128'))
    max_reward, max_reward_sequence = root.evaluate(
        Utarget, actions, reward_dict)
    return max_reward, max_reward_sequence, reward_dict
    

def main():
    max_r, max_r_seq, _ = evaluate_from_root()
    print(f'Max reward was {max_r}\nSequence:\t{max_r_seq}')


if __name__ == '__main__':
    main()
