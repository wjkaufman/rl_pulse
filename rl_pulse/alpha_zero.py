import numpy as np
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim


class Config(object):
    """All the config information for AlphaZero
    """

    def __init__(self):
        # self-"play"
        self.num_actors = 1
        self.num_sampling_moves = 30
        self.max_moves = 48
        # simulations for MCTS
        self.num_simulations = 100
        # root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        # UCB formulat
        self.pb_c_base = 1e2
        self.pb_c_init = 1.25
        # training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096
        # TODO also weight_decay (1e-4), momentum (.9), learning rate schedule


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
        return sample(self.buffer, batch_size)
    
    def __len__(self):
        return(len(self.buffer))


class Policy(nn.Module):
    def __init__(self, input_size=6, lstm_size=16, output_size=5):
        super(Policy, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_size,
                            num_layers=1,
                            batch_first=True)
        self.fc1 = nn.Linear(lstm_size, output_size)
    
    def forward(self, x, h0=None, c0=None):
        """Calculates the policy from state x
        
        Args:
            x: The state of the pulse sequence. Either a tensor with
                shape B*T*(num_actions + 1), or a packed sequence of states.
        """
        if h0 is None or c0 is None:
            x, (h, c) = self.lstm(x)
        else:
            x, (h, c) = self.lstm(x, (h0, c0))
        if type(x) is torch.Tensor:
            x = x[:, -1, :]
        elif type(x) is nn.utils.rnn.PackedSequence:
            # x is PackedSequence, need to get last timestep from each
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            idx = (
                lengths.long() - 1
            ).view(
                -1, 1
            ).expand(
                len(lengths), x.size(2)
            ).unsqueeze(1)
            x = x.gather(1, idx).squeeze(1)
        x = F.softmax(self.fc1(x), dim=1)
        return x, (h, c)


class Value(nn.Module):
    def __init__(self, input_size=6, lstm_size=16):
        super(Value, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_size,
                            num_layers=1,
                            batch_first=True)
        self.fc1 = nn.Linear(lstm_size, 1)
    
    def forward(self, x, h0=None, c0=None):
        """Calculates the value from state x
        
        Args:
            x: The state of the pulse sequence. Either a tensor with
                shape B*T*(num_actions + 1), or a packed sequence of states.
        """
        if h0 is None or c0 is None:
            x, (h, c) = self.lstm(x)
        else:
            x, (h, c) = self.lstm(x, (h0, c0))
        if type(x) is torch.Tensor:
            x = x[:, -1, :]
        elif type(x) is nn.utils.rnn.PackedSequence:
            # x is PackedSequence, need to get last timestep from each
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            idx = (
                lengths.long() - 1
            ).view(
                -1, 1
            ).expand(
                len(lengths), x.size(2)
            ).unsqueeze(1)
            x = x.gather(1, idx).squeeze(1)
        x = self.fc1(x)
        return x, (h, c)


class Network(object):
    """A simple class that combines the policy and value networks or
    uses a single network with policy and value heads.
    """
    
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value
    
    def inference(self, ps_config):
        """
        Args:
            ps_config: A pulse sequence config object
        
        Returns: A tuple (value, policy) where policy is an array of floats
        """
        state = one_hot_encode(ps_config.sequence,
                               num_classes=ps_config.num_pulses + 1,
                               length=ps_config.max_sequence_length)
        state = state.unsqueeze(0)  # add batch dimension
        p, _ = self.policy(state)
        p = p.squeeze()
        v, _ = self.value(state)
        return (float(v), p.detach().numpy())


def one_hot_encode(sequence, num_classes=6, length=48):
    """Takes a pulse sequence and returns a tensor in one-hot encoding
    Args:
        sequence: A list of integers from 0 to num_classes - 2
            with length T. The final value is reserved for
            the start of the sequence.
    
    Returns: A T*num_classes tensor
    """
    state = torch.Tensor(sequence) + 1
    state = torch.cat([torch.Tensor([0]), state])
    state = F.one_hot(state.long(), num_classes).float()
    return state


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
             network=None):
    """Perform rollouts of pulse sequence and
    backpropagate values through nodes, then select
    action based on visit counts of child nodes.

    When looking at AlphaZero code, the game turns into
    the pulse sequence information (sequence, propagators)

    Args:
        propagators: List of Qobj propagators at root.
        sequence: List of ints, represents pulse sequence.
        sequence_length: Maximum length of pulse sequence.
    """
    root = Node(0)
    evaluate(root, ps_config, network=network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        sim_config = ps_config.clone()
        search_path = [node]

        while node.has_children():
            pulse, node = select_child(config, node)
            search_path.append(node)
            sim_config.apply(pulse)

        value = evaluate(node, sim_config, network=network)
        backpropagate(search_path, value)

    return select_action(config, root), root


def evaluate(node, ps_config, network=None):
    """Calculate value and policy predictions from
    the network, add children to node, and return value.
    """
    if network:
        value, policy = network.inference(ps_config)
    else:
        value = 0
        policy = np.ones((ps_config.num_pulses,)) / ps_config.num_pulses
    if ps_config.is_done():
        value = ps_config.value()
    else:
        value = 0  # replace with NN prediction
    valid_pulses = ps_config.get_valid_pulses()
    if len(valid_pulses) > 0:
        for i, p in enumerate(valid_pulses):
            if p not in node.children:
                node.children[p] = Node(policy[i])
    else:
        # no valid pulses, want to avoid this node in the future
        value = -1
    return value


def add_exploration_noise(config, node):
    pulses = list(node.children.keys())
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(pulses))
    frac = config.root_exploration_fraction
    for p, n in zip(pulses, noise):
        node.children[p].prior = node.children[p].prior * (1 - frac) + n * frac


def select_child(config, node):
    """
    """
    _, pulse, child = max(
        (ucb_score(config, node, node.children[pulse]),
         pulse, node.children[pulse])
        for pulse in node.children
    )
    return pulse, child


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


def select_action(config, root):
    visit_counts = [
        root.children[p].visit_count
        for p in root.children
    ]
    probabilities = np.array(visit_counts) / np.sum(visit_counts)
    pulses = list(root.children.keys())
    if len(pulses) == 0:
        raise Exception("Can't select action: no child actions to perform!")
    pulse = np.random.choice(pulses, p=probabilities)
    return pulse


def make_sequence(config, ps_config, network=None):
    """Start with no pulses, do MCTS until a sequence of length
    sequence_length is made.
    """
    search_statistics = []
    while not ps_config.is_done():
        pulse, root = run_mcts(config, ps_config, network=network)
        # print(f'applying pulse {pulse}')
        probabilities = np.zeros((5,))
        for p in root.children:
            probabilities[p] = root.children[p].visit_count / root.visit_count
        # probabilities = [
        #     (p, root.children[p].visit_count / root.visit_count)
        #     for p in root.children
        # ]
        search_statistics.append(
            (ps_config.sequence.copy(),
             probabilities)
        )
        ps_config.apply(pulse, update_propagators=True)
    value = ps_config.value()
    search_statistics = [
        stat + (value, ) for stat in search_statistics
    ]
    return search_statistics
