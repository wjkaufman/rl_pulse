#!/usr/bin/env python
# coding: utf-8

# # Pulse sequence design using evolutionary algorithms
# _Written by Will Kaufman, November 2020_
#
# This notebook tries to replicate results found in Pai Peng et. al.'s preprint.
#
# **TODO** fill in this introduction more!
#
# # TODO
#
# - [ ] see if multiprocessing works `if __name__ == '__main__'`
# - [ ] run!

# In[5]:


import numpy as np
import os
import qutip as qt
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
import datetime

# ## Define algorithm hyperparameters
#
#

# In[328]:


num_actions = 5
population_size = 200
num_generations = 100
# TODO eventually fill in hyperparameters at top of doc


# In[442]:


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(os.path.join(
    'logs', current_time, 'rewards'
)):
    os.makedirs(os.path.join(
        'logs', current_time, 'rewards'
    ))


# ## Initialize the spin system
#
# This sets the parameters of the system ($N$ spin-1/2 particles, which corresponds to a Hilbert space with dimension $2^N$). For the purposes of simulation, $\hbar \equiv 1$.
#
# The total internal Hamiltonian is given by
# $$
# H_\text{int} = C H_\text{dip} + \sum_i^N \delta_i I_z^{i}
# $$
# where $C$ is the coupling strength, $\delta$ is the chemical shift strength (each spin is assumed to be identical), and $H_\text{dip}$ is given by
# $$
# H_\text{dip} = \sum_{i,j}^N d_{i,j} \left(3I_z^{i}I_z^{j} - \mathbf{I}^{i} \cdot \mathbf{I}^{j}\right)
# $$
#
# The target unitary transformation is a simple $\pi/2$-pulse about the x-axis
# $$
# U_\text{target} = \exp\left(-i \frac{\pi}{4} \sum_j I_x^j \right)
# $$
#
# <!-- Hamiltonian is set to be the 0th-order average Hamiltonian from the WHH-4 pulse sequence, which is designed to remove the dipolar interaction term from the internal Hamiltonian. The pulse sequence is $\tau, \overline{X}, \tau, Y, \tau, \tau, \overline{Y}, \tau, X, \tau$.
# The zeroth-order average Hamiltonian for the WAHUHA pulse sequence is
# $$
# H_\text{WHH}^{(0)} = \delta / 3 \sum_i^N \left( I_x^{i} + I_y^{i} + I_z^{i} \right)
# $$ -->

# In[89]:


def make_system(N=3, ):
    # chemical_shifts = np.random.normal(scale=50, size=(N,))
    # Hcs = sum(
    #     [qt.tensor(
    #         [qt.identity(2)]*i
    #         + [chemical_shifts[i] * qt.sigmaz()]
    #         + [qt.identity(2)]*(N-i-1)
    #     ) for i in range(N)]
    # )
    dipolar_matrix = np.random.normal(scale=50, size=(N, N))
    Hdip = sum([
        dipolar_matrix[i, j] * (
            2 * qt.tensor(
                [qt.identity(2)]*i
                + [qt.sigmaz()]
                + [qt.identity(2)]*(j-i-1)
                + [qt.sigmaz()]
                + [qt.identity(2)]*(N-j-1)
            )
            - qt.tensor(
                [qt.identity(2)]*i
                + [qt.sigmax()]
                + [qt.identity(2)]*(j-i-1)
                + [qt.sigmax()]
                + [qt.identity(2)]*(N-j-1)
            )
            - qt.tensor(
                [qt.identity(2)]*i
                + [qt.sigmay()]
                + [qt.identity(2)]*(j-i-1)
                + [qt.sigmay()]
                + [qt.identity(2)]*(N-j-1)
            )
        )
        for i in range(N) for j in range(i+1, N)
    ])
    # Hsys = Hcs + Hdip
    Hsys = Hdip
    X = sum(
        [qt.tensor(
            [qt.identity(2)]*i
            + [qt.sigmax()]
            + [qt.identity(2)]*(N-i-1)
        ) for i in range(N)]
    )
    Y = sum(
        [qt.tensor(
            [qt.identity(2)]*i
            + [qt.sigmay()]
            + [qt.identity(2)]*(N-i-1)
        ) for i in range(N)]
    )
    return Hsys, X, Y


# In[90]:


# Hsys, X, Y = make_system()


# Define the actions as a list of propagators.

# In[277]:


def make_actions(Hsys, X, Y, tau=5e-6, pulse_length=1e-7):
    actions = [
        # delay
        qt.propagator(Hsys, tau),
        # rotations
        qt.propagator(Hsys, tau) * qt.propagator(X, np.pi / 4),
        qt.propagator(Hsys, tau) * qt.propagator(Y, np.pi / 4),
        qt.propagator(Hsys, tau) * qt.propagator(X, -np.pi / 4),
        qt.propagator(Hsys, tau) * qt.propagator(Y, -np.pi / 4)
    ]
    return actions


# In[278]:


# actions = make_actions(Hsys, X, Y)


# ## Define actor and critic networks
#
# The observations of the system are sequences of control amplitudes that have been performed on the system (which most closely represents the knowledge of a typical experimental system). Both the actor and the critic (value) networks share an LSTM layer to convert the sequence of control amplitudes to a hidden state, and two dense layers. Separate policy and value "heads" are used for the two different networks.

# In[38]:


def make_actor(num_actions=5):
    stateful_lstm = tf.keras.layers.LSTM(64, stateful=True)
    hidden1 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
    hidden2 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
    policy = tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.softmax)
    
    actor = tf.keras.models.Sequential([
        stateful_lstm,
        hidden1,
        hidden2,
        policy
    ])
    actor.build(input_shape=(1, None, num_actions))
    return actor


# In[43]:


# obs = tf.reshape(
#     tf.constant(
#         [0,0,1,0,0]
#     ), (1, 1, 5)
# )

# actor = make_actor()
# actor(obs)


# In[419]:


def evaluate_actor(actor, sequence_length=6):
    actor.reset_states()
    Hsys, X, Y = make_system()
    actions = make_actions(Hsys, X, Y,
                           tau=5e-06, pulse_length=1e-07)
    propagator = qt.identity(Hsys.dims[0])
    action = tf.zeros((1, 1, 5))
    for _ in range(sequence_length):
        # determine next action
        probs = actor(action)
        action_ind = tf.squeeze(tf.random.categorical(
            tf.math.log(probs),
            1
        ))
        action = tf.reshape(
            tf.one_hot(
                action_ind, num_actions, 1, 0),
            shape=(1, 1, num_actions))
        # apply next action
        propagator = propagator * actions[action_ind]
    # evaluate fidelity/reward
    target = qt.identity(Hsys.dims[0])
    fidelity = np.clip(np.abs(
        (propagator.dag() * target).tr()
        / qt.identity(Hsys.dims[0]).tr()
    ), 0, 1)
    reward = -np.log10(1 - fidelity + 1e-50)
    return reward


# In[203]:


# rewards = [evaluate_actor(actor) for _ in range(100)]
# plt.hist(rewards)


# In[420]:


def evaluate_actor_mean(actor, sequence_length=6, n=10):
    rewards = [
        evaluate_actor(
            actor,
            sequence_length=sequence_length)
        for _ in range(n)]
    return np.nanmean(rewards)


# In[425]:


def mutate_actor(actor, strength=0.1, fraction=0.25):
    weights = actor.get_weights()
    new_weights = []
    for w in weights:
        shape = w.shape
        ind = np.random.random(size=shape) < fraction
        new_w = w * (1
                     + ind
                     * np.random.normal(scale=strength, size=shape))
        new_weights.append(new_w)
    actor.set_weights(new_weights)


# ## Run EA

# In[430]:


actors = [make_actor() for _ in range(population_size)]


def iterate_population(actors, num_elite=5, num_replace=20):
    """
    Args:
        num_elite: Number of best-performing actors that shouldn't
            be modified.
        num_replace: Number of worst-performing actors that should
            be replaced by copies of other actors.
    """
    description = ''
    rewards = {}
    # evaluate population
    # with ProcessPoolExecutor() as pool:
    #     for actor, reward in enumerate(pool.map(evaluate_actor_mean, actors)):
    #         rewards[actor] = reward
    for actor, reward in zip(actors, map(evaluate_actor_mean, actors)):
        rewards[actor] = reward
    # sort based on performance (best to worst)
    actors = sorted(actors, key=lambda a: rewards[a], reverse=True)
    rewards_list = list(rewards.values())
    new_order = sorted(range(len(actors)),
                       key=lambda a: rewards_list[a], reverse=True)
    rewards = sorted(rewards_list, reverse=True)
    description += ('new order of actors:\t'
                    + ', '.join([str(num) for num in new_order])
                    + '\n')
    # replace worst-performing actors
    actors[(-num_replace):] = [make_actor() for _ in range(num_replace)]
    description += 'actors copied:\t'
    copied_ind = []
    for i in range(num_replace):
        copy_ind = np.random.choice(len(actors) - num_replace)
        copied_ind.append(copy_ind)
        actors[-(i+1)].set_weights(actors[copy_ind].get_weights())
    description += ', '.join([str(num) for num in copied_ind]) + '\n'
    # mutate non-elite actors
    for actor in actors[num_elite:]:
        mutate_actor(actor)
    return actors, rewards, description


# ## Run the EA

# In[441]:


for _ in range(num_generations):
    print(f'on generation {_}')
    actors, rewards, description = iterate_population(actors)
    np.savetxt(os.path.join(
        'logs', current_time, f'rewards/rewards-{_:05.0f}.txt'
    ), rewards)
    if _ % 10 == 0:
        for i, actor in enumerate(actors[:5]):
            actor.save_weights(os.path.join(
                'logs', current_time, 'model', f'actor-{i}-{_:05.0f}'
            ))
    print(description)
