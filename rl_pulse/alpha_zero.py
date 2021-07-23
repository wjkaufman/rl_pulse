import numpy as np
import random
import sys
import os
from functools import lru_cache
import qutip as qt

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath('.'))

import pulse_sequences as ps

NUM_ACTIONS = 6


class Config(object):
    """All the config information for AlphaZero
    """

    def __init__(self):
        # TODO a lot of these aren't needed
        # TODO probably refactor to remove this class entirely...
        # self-"play"
        self.num_actors = 1
        # self.num_sampling_moves = 30
        # self.max_moves = 48
        # simulations for MCTS
        self.num_simulations = 500
        # root prior exploration noise
        self.root_dirichlet_alpha = 2
        self.root_exploration_fraction = 0.25
        # UCB formula
        self.pb_c_base = 1e3
        self.pb_c_init = 1.25
        # training
        # self.training_steps = int(700e3)
        # self.checkpoint_interval = int(1e3)
        # self.window_size = int(1e6)
        # self.batch_size = 4096


class Node(object):
    """A node of the pulse sequence tree. Each node has a particular
    sequence of pulses applied so far.
    """

    def __init__(
            self,
            prior
    ):
        """Create a node at a given point in the pulse sequence.

        Args:
            prior: Prior probability of selecting node.
        """
        self.prior = prior
        self.children = {}
        self.max_value = -1  # maximum value it's seen at any point
        self.visit_count = 0
        self.total_value = 0

    def value(self):
        if self.visit_count > 0:
            return self.total_value / self.visit_count
        else:
            return 0

    def has_children(self):
        return len(self.children) > 0


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, data):
        """Save to replay buffer
        """
        if len(self) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        if batch_size > len(self):
            raise ValueError(f'batch_size of {batch_size} should be'
                             + f'less than buffer size of {len(self)}')
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Network(nn.Module):
    """A network with policy and value heads
    
    TODO Apparently normalization layers don't play nicely
    with multiprocessing. I'll try it without normalization
    to start, but might be worthwhile to investigate later...
    """
    
    def __init__(self,
                 input_size=7,
                 rnn_size=64,
                 fc_size=32,
                 policy_output_size=NUM_ACTIONS,
                 value_output_size=1):
        """
        Args:
            input_size (int): Size of input tensor along feature dimension. If
                the input state is a sequence of actions, should be 5 actions +
                1 start token = 6. If the input state is the sequence of
                toggled spin orientations, should be 6 orientations + 1 start =
                7.
            rnn_size (int): Size of hidden rnn cell.
            fc_size (int): Size of fully connected layer.
            policy_output_size (int): Size of policy output. If output is
                action, should be 5 actions. If output is next orientation of
                spin, should be 6 (where one orientation is invalid with only
                pi/2 pulses).
            value_output_size (int): Size of value output. Should be 1.
        """
        super(Network, self).__init__()
        # define layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=rnn_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        # self.norm1 = nn.BatchNorm1d(rnn_size)
        # self.norm2 = nn.BatchNorm1d(fc_size)
        # self.norm3 = nn.BatchNorm1d(fc_size)
        self.fc1 = nn.Linear(rnn_size, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.fc4 = nn.Linear(fc_size, fc_size)
        self.fc5 = nn.Linear(fc_size, fc_size)
        self.fc6 = nn.Linear(fc_size, fc_size)
        self.fc7 = nn.Linear(fc_size, fc_size)
        self.fc8 = nn.Linear(fc_size, fc_size)
        self.fc9 = nn.Linear(fc_size, fc_size)
        self.fc10 = nn.Linear(fc_size, fc_size)
        self.policy = nn.Linear(fc_size, policy_output_size)
        self.value = nn.Linear(fc_size, value_output_size)

    def forward(self, x, h_0=None):
        """Calculates the policy and value from state x

        Args:
            x: The state of the pulse sequence. Either a tensor with
                shape B*T*input_size, or a packed sequence of states.
        """
        # RNN layer
        if h_0 is None:
            x, h = self.gru(x)
        else:
            x, h = self.gru(x, h_0)
        if type(x) is torch.Tensor:
            x = x[:, -1, :]
        elif type(x) is nn.utils.rnn.PackedSequence:
            # x is PackedSequence, need to get last time step from each
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            idx = (
                lengths.long() - 1
            ).view(-1, 1).expand(
                len(lengths), x.size(2)
            ).unsqueeze(1)
            x = x.gather(1, idx).squeeze(1)
        x = F.relu(x)
        # hidden residual layers
        x = F.relu(self.fc1(x))
        # skip connection from '+ x'
        y = F.relu(self.fc2(x))
        # adding additional layers with skip connections
        x = F.relu((self.fc3(y)) + x)
        y = F.relu(self.fc4(x))
        x = F.relu((self.fc5(y)) + x)
        y = F.relu(self.fc6(x))
        y = F.relu(self.fc7(y))
        x = F.relu((self.fc8(y)) + x)
        # value head
        v = F.relu(self.fc9(x))
        v = F.relu(self.fc10(v))
        value = self.value(v)
        # policy head
        policy = F.softmax(self.policy(x), dim=1)
        return policy, value, h
    
    def save(self):
        # """Save the policy and value networks to a specified path.
        # """
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # torch.save(self.policy.state_dict(), os.path.join(path, 'policy'))
        # torch.save(self.value.state_dict(), os.path.join(path, 'value'))
        raise NotImplementedError()


def one_hot_encode(sequence, num_classes=NUM_ACTIONS+1, start=True):
    """Takes a pulse sequence and returns a tensor in one-hot encoding
    Args:
        sequence: A list of integers from 0 to num_classes - 2
            with length T. The value 0 is reserved for the start
            of the sequence.
        num_classes (int): Number of classes used in one-hot encoding. Includes
            the start token.
        start (bool): If True, add a start token to the start of the sequence.

    Returns: A T*num_classes tensor.
    """
    state = torch.tensor(sequence) + 1
    if start:
        state = torch.cat([torch.tensor([0]), state])
    state = F.one_hot(state.long(), num_classes).float()
    return state


def encode_F_matrix(F, start=True):
    """Encode an F matrix in ML-friendly format. Instead of having a
        3xN matrix, the encoding puts it into a 7xN matrix where the
        first row is a start token, rows 2-4 are +x, +y, +z,
        and 5-7 are -x, -y, -z. All entries are either 0 or 1.
    
    Args:
        F (numpy array): F matrix
        start (bool): If True, adds a start token at the start of sequence.
    
    Returns: A tensor.
    """
    if start:
        encoding = np.zeros((7, F.shape[1] + 1), int)
        encoding[0, 0] = 1
        offset = 1
    else:
        encoding = np.zeros((7, F.shape[1]), int)
        offset = 0
    encoding[1, np.where(F[0, :] == 1)[0] + offset] = 1
    encoding[2, np.where(F[1, :] == 1)[0] + offset] = 1
    encoding[3, np.where(F[2, :] == 1)[0] + offset] = 1
    encoding[4, np.where(F[0, :] == -1)[0] + offset] = 1
    encoding[5, np.where(F[1, :] == -1)[0] + offset] = 1
    encoding[6, np.where(F[2, :] == -1)[0] + offset] = 1
    return torch.tensor(encoding).float()


def pad_and_pack(states):
    """
    Args:
        states: List of variable-length tensors
    """
    lengths = [s.size(0) for s in states]
    return nn.utils.rnn.pack_padded_sequence(
        nn.utils.rnn.pad_sequence(states, batch_first=True),
        lengths, enforce_sorted=False, batch_first=True
    )


def run_mcts(config,
             ps_config,
             network=None, rng=None, sequence_funcs=None,
             test=False):
    """Perform rollouts of pulse sequence and
    backpropagate values through nodes, then select
    action based on visit counts of child nodes.

    When looking at AlphaZero code, the game turns into
    the pulse sequence information (sequence, propagators)
    """
    root = Node(0)
    evaluate(root, ps_config, network=network, sequence_funcs=sequence_funcs)
    add_exploration_noise(config, root, rng=rng)

    for _ in range(config.num_simulations):
        node = root
        sim_config = ps_config.clone()
        search_path = [node]

        while node.has_children():
            action, node = select_child(config, node)
            search_path.append(node)
            sim_config.apply(action)

        value = evaluate(node, sim_config, network=network,
                         sequence_funcs=sequence_funcs)
        backpropagate(search_path, value)

    return select_action(config, ps_config, root, rng=rng, test=test), root


def evaluate(node, ps_config, network=None, sequence_funcs=None):
    """Calculate value and policy predictions from
    the network, add children to node, and return value.
    """
    sequence_tuple = tuple(ps_config.sequence)
    if sequence_funcs is not None:
        (
            get_frame,
            get_reward,
            get_valid_actions,
            get_inference
        ) = sequence_funcs
    else:
        raise Exception('No sequence functions passed!')
    if ps_config.is_done():
        # don't check if pulse sequence is cyclic, just get reward
        value = get_reward(sequence_tuple)
        # # check if pulse sequence is cyclic
        # if (get_frame(sequence_tuple) == np.eye(3)).all():
        #     value = get_reward(sequence_tuple)
        # else:
        #     value = -0.5
    else:
        # pulse sequence is not done yet, estimate value and add children
        if network:
            policy, value, _ = get_inference(sequence_tuple)
        else:
            value = 0
            policy = np.ones((ps_config.num_pulses,)) / ps_config.num_pulses
        valid_actions = get_valid_actions(sequence_tuple)
        # renormalize priors according to valid actions
        policy_renormalized = np.zeros_like(policy)
        policy_renormalized[valid_actions] = policy[valid_actions]
        policy = policy_renormalized / policy_renormalized.sum()
        if len(valid_actions) > 0:
            for p in valid_actions:
                if p not in node.children:
                    node.children[p] = Node(policy[p])
        else:
            # no valid pulses to continue sequence,
            # want to avoid this node in the future
            value = -1
    return value


def add_exploration_noise(config, node, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pulses = list(node.children.keys())
    noise = rng.dirichlet([config.root_dirichlet_alpha] * len(pulses))
    frac = config.root_exploration_fraction
    for p, n in zip(pulses, noise):
        node.children[p].prior = node.children[p].prior * (1 - frac) + n * frac


def select_child(config, node):
    """
    """
    _, action, child = max(
        (ucb_score(config, node, node.children[action]),
         action, node.children[action])
        for action in node.children
    )
    return action, child


def ucb_score(config, parent, child):
    pb_c = np.log10((parent.visit_count + config.pb_c_base + 1)
                    / config.pb_c_base) + config.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


def backpropagate(search_path, value):
    """Propagate value to each node in search path,
    and increment visit counts by 1.
    """
    for node in search_path:
        node.total_value += value
        if value > node.max_value:
            node.max_value = value
        node.visit_count += 1


def select_action(config, ps_config, root, rng=None, test=False):
    """Select an action from root node according to distribution
    of child visit counts (prefer exploration).
    
    Args:
        test (bool): If True, picks the max-probability action.
    """
    if rng is None:
        rng = np.random.default_rng()
    visit_counts = np.zeros(ps_config.num_pulses)
    for p in root.children:
        visit_counts[p] = root.children[p].visit_count
    if np.sum(visit_counts) == 0:
        # raise Exception("Can't select action: no child actions to perform!")
        return None
    probabilities = visit_counts / np.sum(visit_counts)
    pulses = np.arange(ps_config.num_pulses)
    if not test:
        action = rng.choice(pulses, p=probabilities)
    else:
        action = pulses[np.argmax(probabilities)]
    return action


def make_sequence(config, ps_config, network=None, rng=None, test=False,
                  enforce_aht_0=False, max_difference=96, refocus_every=60):
    """Start with no pulses, do MCTS until a sequence of length
    sequence_length is made.
    
    Args:
        test (bool): If True, always picks the max-probability action
            (instead of picking the next action weighted by visit count).
        enforce_aht_0 (bool): If True, require that equal time is spent
            on each axis to satisfy lowest order average Hamiltonian.
        max_difference (int): What is the maximum difference in
            time spent on each axis? If 1, then all interactions
            must be refocused every 6 tau.
        refocus_every (int): How often should interactions be refocused?
            Should be a multiple of 6.
    """
    cache_size = 1000
    
    @lru_cache(maxsize=cache_size)
    def get_frame(sequence):
        if len(sequence) == 0:
            return np.eye(3)
        else:
            return ps.rotations[sequence[-1]] @ get_frame(sequence[:-1])
    
    @lru_cache(maxsize=cache_size)
    def get_axis_counts(sequence):
        if len(sequence) == 0:
            return np.zeros((6,))
        else:
            counts = get_axis_counts(sequence[:-1]).copy()
            frame = get_frame(sequence)
            axis = np.where(frame[-1, :])[0][0]
            is_negative = np.sum(frame[-1, :]) < 0
            counts[axis + 3 * is_negative] += 1
            return counts
    
    @lru_cache(maxsize=cache_size)
    def get_F_matrix(sequence):
        """Get F matrix, as defined in Choi 2020.
        
        Args:
            sequence (tuple): Pulse sequence
        
        """
        if len(sequence) == 0:
            return np.zeros((3, 0), int)
        else:
            frame = get_frame(sequence)
            axis = np.where(frame[-1, :])[0][0]
            F = np.zeros((3, len(sequence)), int)
            F[:, :-1] = get_F_matrix(sequence[:-1])
            F[axis, -1] = frame[-1, axis]
        return F
    
    @lru_cache(maxsize=cache_size)
    def get_propagators(sequence):
        if len(sequence) == 0:
            return ([qt.identity(ps_config.Utarget.dims[0])]
                    * ps_config.ensemble_size)
        else:
            propagators = get_propagators(sequence[:-1])
            propagators = [prop.copy() for prop in propagators]
            for s in range(ps_config.ensemble_size):
                propagators[s] = (
                    ps_config.pulses_ensemble[s][sequence[-1]]
                    * propagators[s]
                )
            return propagators
    
    @lru_cache(maxsize=cache_size)
    def get_reward(sequence):
        propagators = get_propagators(sequence)
        fidelity = 0
        for s in range(ps_config.ensemble_size):
            fidelity += np.clip(
                qt.metrics.average_gate_fidelity(
                    propagators[s],
                    ps_config.Utarget
                ), 0, 1
            )
        fidelity *= 1 / ps_config.ensemble_size
        reward = -1 * np.log10(1 - fidelity + 1e-200)
        return reward
    
    @lru_cache(maxsize=cache_size)
    def get_valid_actions(sequence):
        """
        Args:
            sequence (tuple): Pulse sequence
        """
        F = get_F_matrix(sequence)
        # base constraint: only allow pi/2 pulses
        axis = np.where(F[:, -1])[0][0]
        sign = F[axis, -1]
        invalid_action = axis + 3 * (sign == 1)
        valid_actions = [i for i in range(NUM_ACTIONS) if i != invalid_action]
        return valid_actions
    
    @lru_cache(maxsize=cache_size)
    def get_inference(sequence):
        network.eval()  # switch network to evaluation mode
        F = get_F_matrix(sequence)
        encoding = encode_F_matrix(F, start=True)
        if len(sequence) == 0:
            state = encoding.unsqueeze(0)
            with torch.no_grad():
                (policy, val, h) = network(state)
        else:
            # get hidden layer output while excluding last input in sequence
            (_, _, h) = get_inference(sequence[:-1])
            state = encoding[:, -1].unsqueeze(0)
            with torch.no_grad():
                # get output using intermediate hidden layer and last input
                (policy, val, h) = network(state, h_0=h)
        policy = policy.squeeze().numpy()
        val = val.squeeze().numpy()
        return policy, val, h
    
    sequence_funcs = (get_frame, get_reward, get_valid_actions, get_inference)
    
    # create random number generator (ensure randomness with multiprocessing)
    if rng is None:
        rng = np.random.default_rng()
    search_statistics = []
    while not ps_config.is_done():
        action, root = run_mcts(config, ps_config, network=network, rng=rng,
                                sequence_funcs=sequence_funcs, test=test)
        probabilities = np.zeros((NUM_ACTIONS,))
        for p in root.children:
            probabilities[p] = root.children[p].visit_count / root.visit_count
        # add state, probabilities to search statistics
        search_statistics.append((
            get_F_matrix(ps_config.sequence).copy(),
            # ps_config.sequence.copy(),
            probabilities
        ))
        if action is not None:
            ps_config.apply(action)
        else:
            break
    if action is None:
        value = -1
    else:
        value = get_reward(tuple(ps_config.sequence))
    search_statistics = [
        stat + (value, ) for stat in search_statistics
    ]
    return search_statistics


def convert_stats_to_tensors(stats):
    """
    Args:
        stats (list): A list of tuples, where each tuple contains...
            state: The state, given by the pulse sequence
            prob: The empirical probabilities observed from MCTS
            value: The computed reward for the complete pulse sequence
    """
    output = []
    for s in stats:
        state = encode_F_matrix(s[0])
        probs = torch.tensor(s[1], dtype=torch.float32)
        value = torch.tensor([s[2]], dtype=torch.float32)
        output.append((state,
                       probs,
                       value))
    return output


def train_step(replay_buffer, policy, policy_optimizer, value, value_optimizer,
               writer, global_step=0, num_iters=1, batch_size=64):
    """
    Args:
        global_step: How many minibatches have already been trained on so far.
    Returns: global_step.
    """
    running_policy_loss = 0
    running_value_loss = 0
    for i in range(num_iters):
        if i % 100 == 99:
            print(f'On iteration {i}...', end=' ')
        # zero gradients
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        # select minibatch from replay_buffer
        minibatch = replay_buffer.sample(batch_size=batch_size)
        states, probabilities, values = zip(*minibatch)
        probabilities = torch.cat(probabilities).view(batch_size, -1)
        values = torch.cat(values).view(batch_size, -1)
        packed_states = pad_and_pack(states)
        # policy optimization
        policy_outputs, __ = policy(packed_states)
        policy_loss = -1 / \
            len(states) * torch.sum(probabilities * torch.log(policy_outputs))
        policy_loss.backward()
        policy_optimizer.step()
        # value optimization
        value_outputs, __ = value(packed_states)
        value_loss = F.mse_loss(value_outputs, values)
        value_loss.backward()
        value_optimizer.step()
        # update running losses
        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()
        if i % 100 == 99:
            print(f'policy loss: {policy_loss:.05f},'
                  + f'value loss: {value_loss:.05f}')
            writer.add_scalar('training_policy_loss',
                              running_policy_loss / 100,
                              global_step=global_step)
            writer.add_scalar('training_value_loss',
                              running_value_loss / 100,
                              global_step=global_step)
            running_policy_loss = 0
            running_value_loss = 0
        global_step += 1
    return global_step
