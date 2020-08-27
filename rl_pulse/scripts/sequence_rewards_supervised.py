#!/usr/bin/env python
# coding: utf-8
#
# # Pulse Sequence Design using PPO
# _Written by Will Kaufman, 2020_
import numpy as np
import spin_simulation as ss
from scipy import linalg
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from environments import spin_sys_discrete


def generate_data(
        # simulation parameters
        N=4,
        dim=2**4,
        delta=500,
        coupling=1e3,
        delay=10e-6,
        pulse_width=1e-6,
        # data parameters
        num_configurations=100,
        log_configuration=10,
        num_sequences_per_configuration=100,
        sequence_length=24,  # multiple of 6
        output_path='.',
        filename='data.npz'
        ):
    # TODO define generator for data (i.e. use `yield` statements)?
    
    (X, Y, Z) = ss.get_total_spin(N, dim)
    H_target = ss.get_H_WHH_0(X, Y, Z, delta)
    
    actions = []  # list of TxA tensors for actions
    rewards = []  # list of Tx1 tensors for rewards
    
    for _ in range(num_configurations):
        if _ % log_configuration == 0:
            print(f'on configuration {_}')
        _, Hint = ss.get_H(N, dim, coupling, delta)
        
        Utarget_step = ss.get_propagator(H_target, (pulse_width + delay))
        
        # define actions
        Udelay = linalg.expm(-1j * Hint * delay)
        Ux = linalg.expm(-1j * (X*np.pi/2 + Hint*pulse_width))
        Uxbar = linalg.expm(-1j * (X*-np.pi/2 + Hint*pulse_width))
        Uy = linalg.expm(-1j * (Y*np.pi/2 + Hint*pulse_width))
        Uybar = linalg.expm(-1j * (Y*-np.pi/2 + Hint*pulse_width))
        Ux = Udelay @ Ux
        Uxbar = Udelay @ Uxbar
        Uy = Udelay @ Uy
        Uybar = Udelay @ Uybar
        Udelay = linalg.expm(-1j*(Hint*(pulse_width + delay)))
        action_unitaries = [Ux, Uxbar, Uy, Uybar, Udelay]
        
        for s in range(num_sequences_per_configuration):
            sequence = []
            sequence_rewards = []
            
            Uexp = np.eye(dim, dtype=np.complex64)
            Utarget = np.copy(Uexp)
            
            for t in range(sequence_length):
                action_ind = np.random.choice(5)
                action = np.array(np.arange(5) == action_ind,
                                  dtype=np.int8)
                sequence.append(action)
                Utarget = Utarget_step @ Utarget
                Uexp = action_unitaries[action_ind] @ Uexp
                f = ss.fidelity(Utarget, Uexp)
                r = -np.log10(1 - f + 1e-100)
                sequence_rewards.append(r)
            
            actions.append(np.stack(sequence))
            rewards.append(np.stack(sequence_rewards).astype(np.float16))
        
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    
    # check if directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    np.savez(
        os.path.join(output_path, filename),
        actions=actions,
        rewards=rewards
    )


def load_data(filename):
    
    npz_file = np.load(filename)
    
    actions = tf.convert_to_tensor(npz_file['actions'], dtype=tf.float32)
    rewards = tf.convert_to_tensor(npz_file['rewards'], dtype=tf.float32)
    
    num = np.size(actions, axis=0)
    num_train = int(num * 0.8)
    
    actions_train = actions[:num_train, ...]
    rewards_train = rewards[:num_train, ...]
    actions_val = actions[num_train:, ...]
    rewards_val = rewards[num_train:, ...]
    
    return actions_train, rewards_train, actions_val, rewards_val


def train_eval(
        actions_train,
        rewards_train,
        actions_val,
        rewards_val,
        output_path,
        batch_size=64,
        epochs=50,
        ):
    
    model = keras.Sequential(
        [
            layers.LSTM(64, return_sequences=True, input_shape=(24, 5)),
            layers.Dense(1)
        ]
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsolutePercentageError()]
    )
    
    model.fit(
        actions_train,
        rewards_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(actions_val, rewards_val),
        callbacks=[keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_path, 'logs')
        )]
    )
    
    model.save(os.path.join(output_path, 'my_model'))


def main():
    output_path = os.path.join(
        '..',
        'data',
        datetime.now().strftime("%Y-%m-%d-%H%M%S")
    )
    generate_data(output_path=output_path)
    
    (actions_train, rewards_train, actions_val, rewards_val) = \
        load_data(os.path.join(output_path, 'data.npz'))
    train_eval(
        actions_train, rewards_train,
        actions_val, rewards_val,
        output_path
    )


if __name__ == '__main__':
    main()
