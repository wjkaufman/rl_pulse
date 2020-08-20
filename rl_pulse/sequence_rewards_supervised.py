#!/usr/bin/env python
# coding: utf-8
#
# # Pulse Sequence Design using PPO
# _Written by Will Kaufman, 2020_
import numpy as np
import spin_simulation as ss
from scipy import linalg

import tensorflow as tf

# from environments import spin_sys_discrete


def generate_training_data(
        N=4,
        dim=2**4,
        delta=500,
        coupling=1e3,
        delay=10e-6,
        pulse_width=500
        ):
    # TODO define generator for data (i.e. use `yield` statements)?
    
    (X, Y, Z) = ss.get_total_spin(N, dim)
    H_target = ss.get_H_WHH_0(X, Y, Z, delta)
    
    actions = []  # list of TxA tensors for actions
    rewards = []  # list of Tx1 tensors for rewards
    
    for _ in range(1000):
        if _ % 100 == 0:
            print(f'on iteration {_}')
        _, Hint = ss.get_H(N, dim, coupling, delta)
        
        Utarget_step = ss.get_propagator(H_target, (pulse_width + delay))
        Uexp = np.eye(dim, dtype=np.complex128)
        Utarget = np.copy(Uexp)
        
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
        action_unitaries = [Ux, Uxbar, Uy, Uybar, Udelay]
        
        sequence = []
        rewards = []
        
        for t in range(24):
            action_ind = np.random.choice(5)
            action = tf.convert_to_tensor(np.arange(5) == action_ind,
                                          dtype=tf.int64)
            sequence.append(action)
            Utarget = Utarget_step @ Utarget
            Uexp = action_unitaries[action_ind] @ Uexp
            f = ss.fidelity(Utarget, Uexp)
            r = -np.log10(1 - f + 1e-100)
            rewards.append(r)
        
        actions.append(tf.stack(sequence))
        rewards.append(tf.convert_to_tensor(rewards))
        
    actions = tf.stack(actions)
    rewards = tf.stack(rewards)
    
    # make dataset
    ds = tf.data.Dataset.from_tensor_slices((actions, rewards))
    print(ds)
    # TODO figure out how to save this???
    filename = 'data.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(ds)
    return ds


def train_eval(
        # parameters here
        ):
    # TODO
    pass


def main():
    ds = generate_training_data()
    print(ds)
    # train_eval()


if __name__ == '__main__':
    main()
