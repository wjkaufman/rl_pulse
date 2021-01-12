#!/usr/bin/env python
# coding: utf-8

import qutip as qt
import numpy as np
import sys
import os
import multiprocessing as mp

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath('..'))

import alpha_zero as az
import pulse_sequences as ps


num_cores = 32
num_collect = 100
batch_size = 64
num_iters = 100

delay = 1e-2  # time is relative to chemical shift strength
pulse_width = 1e-3
N = 3  # number of spins
ensemble_size = 5


Utarget = qt.tensor([qt.identity(2)] * N)


rb = az.ReplayBuffer(int(1e5))


def collect_data_no_net(x):
    print(f'collecting data without network ({x})')
    config = az.Config()
    config.num_simulations = 500
    ps_config = ps.PulseSequenceConfig(N=N, ensemble_size=ensemble_size,
                                       max_sequence_length=48, Utarget=Utarget,
                                       pulse_width=pulse_width, delay=delay)
    return az.make_sequence(config, ps_config, network=None, rng=ps_config.rng)


with mp.Pool(num_cores) as pool:
    output = pool.map(collect_data_no_net, range(num_collect))
for stat in output:
    az.add_stats_to_buffer(stat, rb)


policy = az.Policy()
value = az.Value()
net = az.Network(policy, value)


net.save('network')


policy_optimizer = optim.Adam(policy.parameters(), lr=1e-5)
value_optimizer = optim.Adam(value.parameters(), lr=1e-5)
writer = SummaryWriter()
global_step = 0  # how many minibatches the models have been trained


def collect_data(x):
    print(f'collecting data ({x})')
    config = az.Config()
    config.num_simulations = 250
    ps_config = ps.PulseSequenceConfig(N=N, ensemble_size=ensemble_size,
                                       max_sequence_length=48, Utarget=Utarget,
                                       pulse_width=pulse_width, delay=delay)
    # load policy and value networks from memory
    policy = az.Policy()
    policy.load_state_dict(torch.load('network/policy'))
    policy.eval()
    value = az.Value()
    value.load_state_dict(torch.load('network/value'))
    value.eval()
    net = az.Network(policy, value)
    return az.make_sequence(config, ps_config, network=net, rng=ps_config.rng)


for i in range(10):
    print(f'on iteration {i}')
    # collect data
    print('collecting data...')
    with mp.Pool(num_cores) as pool:
        output = pool.map(collect_data, range(num_collect))
    for stat in output:
        az.add_stats_to_buffer(stat, rb)
    mean_value = np.mean([o[-1][-1] for o in output])
    for o in output:
        if o[-1][-1] > 1:
            print('Candidate pulse sequence found! Value is ',
                  o[-1][-1], '\n', o[-1][0])
    writer.add_scalar('mean_value', mean_value, global_step=global_step)
    # train models from replay buffer
    print('training model...')
    global_step = az.train_step(rb, policy, policy_optimizer,
                                value, value_optimizer,
                                writer, global_step=global_step,
                                num_iters=num_iters, batch_size=batch_size)
    # write updated weights to file
    net.save('network')
