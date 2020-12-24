import numpy as np
# import qutip as qt
# import pulse_sequences as ps


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
    evaluate(root, ps_config)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        sim_config = ps_config.clone()
        search_path = [node]

        while node.has_children():
            pulse, node = select_child(config, node)
            search_path.append(node)
            sim_config.apply(pulse)

        value = evaluate(node, sim_config)
        backpropagate(search_path, value)

    return select_action(config, root), root


def evaluate(node, ps_config, network=None):
    """Calculate value and policy predictions from
    the network, add children to node, and return value.
    """
    if ps_config.is_done():
        value = ps_config.value()
    else:
        value = 0  # replace with NN prediction
    valid_pulses = ps_config.get_valid_pulses()
    if len(valid_pulses) > 0:
        policy = np.ones((len(valid_pulses),)) / len(valid_pulses)
        # TODO replace ^ with NN prediction
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
        print(f'applying pulse {pulse}')
        probabilities = [
            (p, root.children[p].visit_count / root.visit_count)
            for p in root.children
        ]
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
