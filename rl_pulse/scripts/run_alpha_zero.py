import qutip as qt
import sys
import os
from datetime import datetime
import random
from time import sleep
# from copy import deepcopy

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath('..'))

import pulse_sequences as ps
import alpha_zero as az

mp.set_sharing_strategy('file_system')

collect_no_net_procs = 12  # 15
collect_no_net_count = 700  # 700
collect_procs = 12  # 15
collect_count = 1000  # 1000

buffer_size = int(1e6)  # 1e6
batch_size = 2048  # 2048
num_iters = int(800e3)  # 800e3

max_sequence_length = 48

print_every = 10  # 100
save_every = 1000  # 1000


delay = 1e-2  # time is relative to chemical shift strength
pulse_width = 1e-3
N = 3  # number of spins
ensemble_size = 5


Utarget = qt.tensor([qt.identity(2)] * N)


def collect_data_no_net(proc_num, buffer, index, lock, buffer_size, ps_count):
    """
    Args:
        proc_num: Which process number this is (for debug purposes)
        buffer (mp.managers.List): A shared replay buffer
        index (mp.managers.Value): The current index for the buffer
        lock (mp.managers.RLock): Lock object to prevent overwriting
            data from different threads
        buffer_size (int): The maximum size of the buffer
        ps_count (Value): Shared count of how many pulse sequences have
            been constructed
    """
    print(datetime.now(), f'collecting data without network ({proc_num})')
    config = az.Config()
    ps_config = ps.PulseSequenceConfig(N=N, ensemble_size=ensemble_size,
                                       max_sequence_length=max_sequence_length,
                                       Utarget=Utarget,
                                       pulse_width=pulse_width, delay=delay)
    for i in range(collect_no_net_count):
        ps_config.reset()
        output = az.make_sequence(
            config, ps_config, network=None, rng=ps_config.rng)
        if output[-1][2] > 2.5:
            print(datetime.now(),
                  f'candidate pulse sequence from {proc_num}',
                  output[-1])
        output_tensors = az.convert_stats_to_tensors(output)
        with lock:
            ps_count.value += 1
            print(datetime.now(), proc_num, f'ps_count: {ps_count.value}',
                  f'index: {index.value}',
                  f'buffer length: {len(buffer)}, buffer_size: {buffer_size}')
        for obs in output_tensors:
            with lock:
                if len(buffer) < buffer_size:
                    buffer.append(obs)
                else:
                    buffer[index.value] = obs
                index.value += 1
                if index.value >= buffer_size:
                    print(datetime.now(), f'proc {proc_num}, apparently',
                          f'index {index.value} >= buffer size {buffer_size}',
                          flush=True)
                    # index.value = 0
                    # TODO this should break the code, but idk why it's
                    # not working so lol


def collect_data(proc_num, buffer, index, lock, buffer_size, net, ps_count):
    """
    Args:
        ps_count (Value): A shared count of how many pulse sequences have been
            constructed so far
    """
    print(datetime.now(), f'collecting data ({proc_num})')
    config = az.Config()
    config.num_simulations = 250
    ps_config = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                       ensemble_size=ensemble_size,
                                       max_sequence_length=max_sequence_length,
                                       pulse_width=pulse_width, delay=delay)
    for _ in range(collect_count):
        ps_config.reset()
        output = az.make_sequence(
            config, ps_config, network=net, rng=ps_config.rng)
        if output[-1][2] > 2:
            print(datetime.now(),
                  f'candidate pulse sequence from {proc_num}',
                  output[-1])
        output_tensors = az.convert_stats_to_tensors(output)
        with lock:
            ps_count.value += 1
            print(datetime.now(), proc_num, f'ps_count: {ps_count.value}',
                  f'index: {index.value}',
                  f'buffer length: {len(buffer)}, buffer_size: {buffer_size}')
        for obs in output_tensors:
            with lock:
                if len(buffer) < buffer_size:
                    buffer.append(obs)
                else:
                    buffer[index.value] = obs
                index.value += 1
                if index.value >= buffer_size:
                    print(datetime.now(), f'proc {proc_num} w/net, apparently',
                          f'index {index.value} >= buffer size {buffer_size}',
                          flush=True)
                    # index.value = 0
                    # TODO eventually change this back...


def train_process(proc_num, buffer, lock, net, global_step, ps_count):
    """
    Args:
        buffer (mp.managers.list): Replay buffer,
            list of (state, probability, value).
        global_step (mp.managers.Value): Counter to keep track
            of training iterations
        writer (SummaryWriter): Write losses to log
    """
    print(datetime.now(), f'started training process ({proc_num})')
    writer = SummaryWriter()
    net_optimizer = optim.Adam(net.parameters(),)
    # number of training iterations
    for i in range(num_iters):
        with lock:
            buffer_len = len(buffer)
        if i % save_every == 0:
            print(datetime.now(), 'saving network...')
            # save network parameters to file
            if not os.path.exists('network'):
                os.makedirs('network')
            torch.save(net.state_dict(), f'network/{i:07.0f}-network')
        if buffer_len < batch_size:
            print(datetime.now(), 'not enough data yet, sleeping...')
            sleep(5)
            continue
        elif buffer_len < 1e4:
            sleep(.5)  # put on the brakes a bit, don't tear through the data
        net_optimizer.zero_grad()
        # sample minibatch from replay buffer
        with lock:
            minibatch = random.sample(list(buffer), batch_size)
            # minibatch = deepcopy(minibatch)
        states, probabilities, values = zip(*minibatch)
        probabilities = torch.stack(probabilities)
        values = torch.stack(values)
        packed_states = az.pad_and_pack(states)
        # evaluate network
        policy_outputs, value_outputs, _ = net(packed_states)
        policy_loss = -1 / \
            len(states) * torch.sum(probabilities * torch.log(policy_outputs))
        value_loss = F.mse_loss(value_outputs, values)
        loss = policy_loss + value_loss
        loss.backward()
        net_optimizer.step()
        # write losses to log
        with lock:
            writer.add_scalar('training_policy_loss',
                              policy_loss, global_step=global_step.value)
            writer.add_scalar('training_value_loss',
                              value_loss, global_step=global_step.value)
        # every print_every iterations, save histogram of replay buffer values
        if i % print_every == 0:
            print(datetime.now(), f'updated network (iteration {i})',
                  f'pulse_sequence_count: {ps_count.value}')
            with lock:
                _, _, values = zip(*list(buffer))
                # values = deepcopy(values)
                values = torch.stack(values).squeeze()
                writer.add_histogram('buffer_values', values,
                                     global_step=global_step.value)
                writer.add_scalar('pulse_sequence_count',
                                  ps_count.value, global_step.value)
        with lock:
            global_step.value += 1


if __name__ == '__main__':
    mp.set_start_method('spawn')
    with mp.Manager() as manager:
        buffer = manager.list()
        index = manager.Value(typecode='i', value=0)
        global_step = manager.Value('i', 0)
        ps_count = manager.Value('i', 0)
        lock = manager.Lock()
        # get network
        net = az.Network()
        net.share_memory()
        collectors = []
        for i in range(collect_no_net_procs):
            c = mp.Process(target=collect_data_no_net,
                           args=(i, buffer, index, lock,
                                 buffer_size, ps_count))
            c.start()
            collectors.append(c)
        trainer = mp.Process(target=train_process,
                             args=(4, buffer, lock, net,
                                   global_step, ps_count))
        trainer.start()
        # join collectors before starting more
        for c in collectors:
            c.join()
        print(datetime.now(), 'apparently done with initial collect,'
              + f'ps_count: {ps_count.value},'
              + f'global_step: {global_step.value}')
        collectors.clear()
        # start data collectors with network
        for i in range(collect_procs):
            c = mp.Process(target=collect_data,
                           args=(i, buffer, index, lock,
                                 buffer_size, net, ps_count))
            c.start()
            collectors.append(c)
        for c in collectors:
            c.join()
        print(datetime.now(), 'apparently done with data collection,'
              + f'ps_count: {ps_count.value},'
              + f'global_step: {global_step.value}')
        trainer.join()
        print(datetime.now(), 'trainer is joined')
        print(datetime.now(), 'done!')
