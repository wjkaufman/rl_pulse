from datetime import datetime
import random
from time import sleep
import qutip as qt
import sys
import os
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

sys.path.append(os.path.abspath('..'))

import alpha_zero as az
import pulse_sequences as ps

collect_no_net_procs = 0
collect_no_net_count = 0
collect_procs = 14

buffer_size = int(1e6)
batch_size = 2048
num_iters = int(1e6)

max_sequence_length = 12

print_every = 100
save_every = 500

reward_threshold = 3

dipolar_strength = 1e2
pulse_width = 2e-5
delay = 1e-4
N = 3
ensemble_size = 50
rot_error = 1e-2
phase_transient_error = 1e-4
offset_error = 1e1


Utarget = qt.identity([2] * N)


def collect_data(proc_num, queue, net, ps_count, global_step, lock):
    """
    Args:
        queue (Queue): A queue to add the statistics gathered
            from the MCTS rollouts.
        ps_count (Value): A shared count of how many pulse sequences have been
            constructed so far
    """
    print(datetime.now(), f'collecting data ({proc_num})')
    config = az.Config()
    ps_config = ps.PulseSequenceConfig(
        Utarget=Utarget, N=N,
        ensemble_size=ensemble_size,
        max_sequence_length=max_sequence_length,
        dipolar_strength=dipolar_strength,
        pulse_width=pulse_width, delay=delay,
        rot_error=rot_error,
        phase_transient_error=phase_transient_error,
        offset_error=offset_error
    )
    while global_step.value < num_iters:
        ps_config.reset()
        output = az.make_sequence(config, ps_config, network=net,
                                  rng=ps_config.rng, enforce_aht_0=True,
                                  max_difference=2)  # , refocus_every=6
        if output[-1][2] > reward_threshold:
            print(datetime.now(),
                  f'candidate pulse sequence from {proc_num}',
                  output[-1])
        with lock:
            queue.put(output)
            ps_count.value += 1


def train_process(queue, net, global_step, ps_count, lock,
                  c_value=1e0, c_l2=1e-6):
    """
    Args:
        queue (Queue): A queue to add the statistics gathered
            from the MCTS rollouts.
        global_step (mp.managers.Value): Counter to keep track
            of training iterations
        writer (SummaryWriter): Write losses to log
    """
    start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # create directory to store results in
    if not os.path.exists(start_time):
        os.makedirs(start_time)
    writer = SummaryWriter(start_time)
    net_optimizer = optim.Adam(net.parameters(),)
    
    buffer = []
    index = 0
    i = 0
    
    # a list to keep track of rewards, periodically save hists
    rewards = []
    
    # write network structure to tensorboard file
    tmp = torch.zeros((1, 10, 7))  # TODO don't hard code in number of actions
    writer.add_graph(net, tmp)
    del tmp
    
    while global_step.value < num_iters:
        # get stats from queue
        with lock:
            while not queue.empty():
                # TODO save data on pulse sequence reward distribution
                # Issue #17
                new_stats = queue.get()
                for stat in new_stats:
                    stat_tensor = az.convert_stat_to_tensor(stat)
                    if len(buffer) < buffer_size:
                        buffer.append(stat_tensor)
                    else:
                        buffer[index] = stat
                    index = index + 1 if index < buffer_size - 1 else 0
                    # check if stat is final pulse sequence
                    if stat[0].shape[1] + 1 == max_sequence_length:
                        rewards.append(stat[2])
        
        # check if there's enough stats to start training
        if len(buffer) < batch_size:
            print(datetime.now(), 'not enough data yet, sleeping...')
            sleep(5)
            continue

        if i % save_every == 0:
            print(datetime.now(), 'saving data...')
            
            # saving network
            if not os.path.exists(f'{start_time}/network/'):
                os.makedirs(f'{start_time}/network/')
            torch.save(net.state_dict(),
                       f'{start_time}/network/{i:07.0f}')
            
            # saving reward histogram
            reward_hist = np.histogram(rewards,
                                       bins=np.linspace(0, 10, 101))
            if not os.path.exists(f'{start_time}/reward_hist/'):
                os.makedirs(f'{start_time}/reward_hist/')
            np.savez_compressed(
                f'{start_time}/reward_hist/{i:07.0f}.npz',
                counts=reward_hist[0], bins=reward_hist[1]
            )
            rewards = []

        net_optimizer.zero_grad()

        minibatch = random.sample(buffer, batch_size)
        states, probabilities, values = zip(*minibatch)
        probabilities = torch.stack(probabilities)
        values = torch.stack(values)
        packed_states = az.pad_and_pack(states)

        policy_outputs, value_outputs, _ = net(packed_states)
        policy_loss = -1 / \
            len(states) * torch.sum(probabilities * torch.log(policy_outputs))
        value_loss = F.mse_loss(value_outputs, values)
        l2_reg = torch.tensor(0.)
        for param in net.parameters():
            l2_reg += torch.norm(param)
        loss = policy_loss + c_value * value_loss + c_l2 * l2_reg
        loss.backward()
        net_optimizer.step()

        writer.add_scalar('training_policy_loss',
                          policy_loss, global_step=global_step.value)
        writer.add_scalar('training_value_loss',
                          c_value * value_loss, global_step=global_step.value)
        writer.add_scalar('training_l2_reg',
                          c_l2 * l2_reg, global_step=global_step.value)

        if i % print_every == 0:
            print(datetime.now(), f'updated network (iteration {i})',
                  f'pulse_sequence_count: {ps_count.value}')
            _, _, values = zip(*list(buffer))
            values = torch.stack(values).squeeze()
            writer.add_histogram('buffer_values', values,
                                 global_step=global_step.value)
            writer.add_scalar('pulse_sequence_count', ps_count.value,
                              global_step.value)
        with lock:
            global_step.value += 1
        i += 1
        sleep(.1)


if __name__ == '__main__':
    with mp.Manager() as manager:
        queue = manager.Queue()
        global_step = manager.Value('i', 0)
        ps_count = manager.Value('i', 0)
        lock = manager.Lock()

        net = az.Network()
        # optionally load state dict
        # change global_step above too...
        # net.load_state_dict(torch.load('0026000-network'))
        net.share_memory()
        collectors = []
        for i in range(collect_procs):
            c = mp.Process(target=collect_data,
                           args=(i, queue, net, ps_count, global_step, lock))
            c.start()
            collectors.append(c)
        trainer = mp.Process(target=train_process,
                             args=(queue, net,
                                   global_step, ps_count, lock))
        trainer.start()
        
        for c in collectors:
            c.join()
        collectors.clear()
        
        print('all collectors are joined')
        trainer.join()
        print('trainer is joined')
        print('done!')
