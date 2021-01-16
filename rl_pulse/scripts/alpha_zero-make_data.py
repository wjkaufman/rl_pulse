import qutip as qt
import sys
import os
import multiprocessing as mp

import torch

sys.path.append(os.path.abspath('..'))

import alpha_zero as az
import pulse_sequences as ps


num_cores = 32  # 32
num_collect_initial = 2000  # 5000

max_sequence_length = 48


delay = 1e-2  # time is relative to chemical shift strength
pulse_width = 1e-3
N = 3  # number of spins
ensemble_size = 5


Utarget = qt.tensor([qt.identity(2)] * N)


rb = az.ReplayBuffer(int(1e5))


def collect_data_no_net(x):
    config = az.Config()
    ps_config = ps.PulseSequenceConfig(N=N, ensemble_size=ensemble_size,
                                       max_sequence_length=max_sequence_length,
                                       Utarget=Utarget,
                                       pulse_width=pulse_width, delay=delay)
    return az.make_sequence(config, ps_config, network=None, rng=ps_config.rng)


with mp.Pool(num_cores) as pool:
    output = pool.map(collect_data_no_net, range(num_collect_initial))
for stat in output:
    az.add_stats_to_buffer(stat, rb)


states, probabilities, values = zip(*rb.buffer)


packed_states = az.pad_and_pack(states)
probabilities = torch.cat(probabilities).view(len(rb), -1)
values = torch.cat(values).view(len(rb), -1)


torch.save(packed_states, 'states.pt')
torch.save(probabilities, 'probabilities.pt')
torch.save(values, 'values.pt')
