#!/usr/bin/env python
# coding: utf-8

# # Pulse design using reinforcement learning
# _Written by Will Kaufman, November 2020_


import numpy as np
import os
import sys
import qutip as qt
import tensorflow as tf
import datetime

sys.path.append('../..')  # for running jobs on Discovery

from rl_pulse.environments import spin_system_continuous

# TODO eventually fill in hyperparameters at top of doc
discount = 0.99
stddev = 1e-3


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

# In[ ]:


N = 3  # 4-spin system


# In[ ]:


chemical_shifts = np.random.normal(scale=50, size=(N,))
Hcs = sum(
    [qt.tensor(
        [qt.identity(2)]*i
        + [chemical_shifts[i] * qt.sigmaz()]
        + [qt.identity(2)]*(N-i-1)
    ) for i in range(N)]
)


# In[ ]:


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


# In[ ]:


Hsys = Hcs + Hdip
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
# Z = qt.tensor([qt.sigmaz()]*N)
Hcontrols = [50e3 * X, 50e3 * Y]
target = qt.propagator(X, np.pi/4)


# The `SpinSystemContinuousEnv` simulates the quantum system given above, and exposes relevant methods for RL (including a `step` method that takes an action and returns an observation and reward, a `reset` method to reset the system).

# In[ ]:


env = spin_system_continuous.SpinSystemContinuousEnv(
    Hsys=Hsys,
    Hcontrols=Hcontrols,
    target=target,
    discount=discount
)


# ## Define metrics

# In[ ]:


loss_pg_metric = tf.keras.metrics.Mean('loss_pg', dtype=tf.float32)
loss_vf_metric = tf.keras.metrics.Mean('loss_vf', dtype=tf.float32)
infidelity_metric = tf.keras.metrics.Mean('infidelity', dtype=tf.float32)


# In[ ]:


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join('logs', current_time, 'train')
test_log_dir = os.path.join('logs', current_time, 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
if not os.path.exists(os.path.join(
    'logs', current_time, 'controls'
)):
    os.makedirs(os.path.join(
        'logs', current_time, 'controls'
    ))


# In[ ]:


# count number of time steps played through
global_step = tf.Variable(0, trainable=False, name='global_step')


# ## Define actor and critic networks
#
# The observations of the system are sequences of control amplitudes that have been performed on the system (which most closely represents the knowledge of a typical experimental system). Both the actor and the critic (value) networks share an LSTM layer to convert the sequence of control amplitudes to a hidden state, and two dense layers. Separate policy and value "heads" are used for the two different networks.

# In[ ]:


lstm = tf.keras.layers.LSTM(64)
stateful_lstm = tf.keras.layers.LSTM(64, stateful=True)
hidden1 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
hidden2 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
policy = tf.keras.layers.Dense(2, activation=tf.keras.activations.tanh)
value = tf.keras.layers.Dense(1)


# In[ ]:


actor_net = tf.keras.models.Sequential([
    lstm,
    hidden1,
    hidden2,
    policy
])


# In[ ]:


critic_net = tf.keras.models.Sequential([
    lstm,
    hidden1,
    hidden2,
    value
])


# In[ ]:


stateful_actor_net = tf.keras.models.Sequential([
    stateful_lstm,
    hidden1,
    hidden2,
    policy
])


# ## Define PPO agent
#
# [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) is a state-of-the-art RL algorithm that can be used for both discrete and continuous action spaces. PPO prevents the policy from over-adjusting during training by defining a clipped policy gradient loss function:
# $$
# L^\text{clip}(\theta) = \mathbb{E}_t\left[
# \min(r_t(\theta)\hat{A}_t, \text{clip}(
#     r_t(\theta), 1-\epsilon, 1+\epsilon)
# )\hat{A}_t
# \right]
# $$
# where the "importance ratio" $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ is the relative probability of choosing the action under the new policy compared to the old policy. By clipping the loss function, there is non-zero gradient only in a small region around the original policy.
#
# Because the actor and critic networks share layers, the total loss function is used for training
# $$
# L(\theta) = \mathbb{E}_t \left[
# -L^\text{clip}(\theta) + c_1 L^\text{VF}(\theta)
# \right]
# $$
# with $L^\text{VF}(\theta)$ as the MSE loss for value estimates.
#
# Basing off TF-Agents [abstract base class](https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/TFAgent). Also using [PPOAgent code](https://github.com/tensorflow/agents/blob/v0.6.0/tf_agents/agents/ppo/ppo_agent.py#L746).

# ## Collect some experience from the environment

# The following collects experience by interacting with the environment.
#
# All the data should have dimensions `batch_size * [other dims]`, shouldn't just be `batch_size`.

# In[ ]:


def calculate_returns(rewards, step_types, discounts):
    """
    Args:
        rewards: A tensor of rewards for the episode.
            Should have size batch_size * 1.
        step_types: A tensor of step types, 1 if a normal step.
            Should have size batch_size * 1.
    """
    returns = [0] * rewards.shape[0]
    returns[-1] = rewards[-1]
    for i in range(1, len(rewards)):
        returns[-(i + 1)] = (discounts[-i]
                             * returns[-i]
                             * tf.cast(step_types[-(i + 1)] == 1,
                                       tf.float32)) + rewards[-(i+1)]
    return returns


# In[ ]:


def get_obs_and_mask(obs_list, max_sequence_length=100):
    obs = []
    mask = []
    num_features = obs_list[0].shape[-1]
    for i in range(len(obs_list)):
        obs_length = obs_list[i].shape[-2]
        obs.append(tf.concat(
            [tf.cast(obs_list[i], tf.float32),
             tf.zeros((1,
                       max_sequence_length - obs_length,
                       num_features))],
            axis=1
        ))
        mask.append(tf.concat(
            [tf.ones((1, obs_length)),
             tf.zeros((1,
                       max_sequence_length - obs_length))],
            axis=1
        ))
    obs = tf.squeeze(tf.stack(obs))
    mask = tf.squeeze(tf.stack(mask))
    return obs, mask


# In[ ]:


def collect_experience(
        num_steps=100,
        stddev=1e-3,
        max_sequence_length=100):
    """Collect experience from the environment.
    Starts with a new episode and collects num_steps
    
    Args:
        num_steps: How many steps to collect from the environment.
        stddev: Standard deviation of noise applied to actions.
        max_sequence_length: For sequence data, how long to make
            the max sequence.
    """
    step_types = []
    discounts = []
    observations = []
    actions = []
    action_means = []
    rewards = []
    step = env.reset()
    if stateful_actor_net.built:
        stateful_actor_net.reset_states()
    # collect experience
    for _ in range(num_steps):
        observations.append(step.observation)
        action_mean = stateful_actor_net(
            tf.expand_dims(step.observation[:, -1, :], 1)
        )
        action_means.append(action_mean)
        action = action_mean + tf.random.normal(shape=action_mean.shape,
                                                stddev=stddev)
        actions.append(action)
        step = env.step(action)
        rewards.append(step.reward)
        step_types.append(step.step_type)
        discounts.append(step.discount)
        if step.step_type == 2:
            # episode is done, reset environment and network state
            stateful_actor_net.reset_states()
            infidelity_metric(1 - env.fidelity())
            step = env.reset()
    # put data into tensors
    step_types = tf.stack(step_types)
    discounts = tf.stack(discounts)
    actions = tf.squeeze(tf.stack(actions))
    action_means = tf.squeeze(tf.stack(action_means))
    rewards = tf.stack(rewards)
    # reshape observations to be same sequence length, and create
    # a mask for original sequence length
    obs, mask = get_obs_and_mask(observations,
                                 max_sequence_length=max_sequence_length)
    returns = tf.stack(calculate_returns(rewards, step_types, discounts))
    advantages = returns - critic_net(obs, mask=mask)
    old_action_log_probs = tf.reduce_sum(
        -((actions - action_means) / stddev)**2,
        axis=1,
        keepdims=True
    )
    return (
        obs, mask, actions, action_means,
        rewards, step_types, discounts, returns,
        advantages, old_action_log_probs
    )


# In[ ]:


# %lprun -f collect_experience collect_experience()


# In[ ]:


# (
#     obs, mask, actions, action_means,
#     rewards, step_types, discounts, returns,
#     advantages, old_action_log_probs
# ) = collect_experience(num_steps=500,
#                        stddev=stddev,
#                        max_sequence_length=100)


# ## Evaluate the actor

# In[ ]:


def evaluate_actor():
    """Evaluate the actor.
    
    Returns: A tuple containing
        rewards: A tensor of rewards for the episode.
        control_amplitudes: A tensor of control amplitudes
            applied during the episode.
    """
    rewards = []
    actions = []
    step = env.reset()
    if stateful_actor_net.built:
        stateful_actor_net.reset_states()
    while step.step_type <= 1:
        action = stateful_actor_net(
            tf.expand_dims(step.observation[:, -1, :], 1)
        )
        step = env.step(action)
        actions.append(action)
        rewards.append(step.reward)
    rewards = tf.stack(rewards)
    infidelity_metric(1 - env.fidelity())
    return rewards, tf.squeeze(step.observation)


# In[ ]:


# a, b = evaluate_actor()


# In[ ]:


# plt.plot(a.numpy())


# In[ ]:


# plt.plot(b.numpy()[:,0], label='x')
# plt.plot(b.numpy()[:,1], label='y')
# plt.legend()


# ## Training
#
# To "train" the actor and critic networks (change the network parameters to minimize the loss function), an optimizer using the Adam algorithm is used. The loss function is described above, and is composed of policy-gradient loss and value function loss.

# In[ ]:


optimizer = tf.optimizers.Adam()
mse = tf.losses.mse


# In[ ]:


if not critic_net.built:
    critic_net.build(input_shape=(None, None, 2))
if not actor_net.built:
    actor_net.build(input_shape=(None, None, 2))


# Define a list of trainable variables that should be updated when minimizing the loss function.

# In[ ]:


critic_vars = critic_net.trainable_variables
actor_vars = actor_net.trainable_variables
trainable_variables = set()
for var in critic_vars + actor_vars:
    trainable_variables.add(var.ref())
trainable_variables = list(trainable_variables)
trainable_variables = [var.deref() for var in trainable_variables]


# In[ ]:


def calculate_gradients(
        actor_net,
        critic_net,
        trainable_variables,
        obs,
        mask,
        actions,
        action_means,
        old_action_log_probs,
        returns,
        advantages,
        stddev=1e-3,
        epsilon=.2,
        c1=1):
    """
    Returns: tuple containing
        l: Total loss.
        grad: Gradient of loss wrt trainable variables.
    """
    batch_size = obs.shape[0]
    with tf.GradientTape() as tape:
        action_log_probs = (
            tf.reduce_sum(-((actions - actor_net(obs, mask)) / stddev)**2,
                          axis=1,
                          keepdims=True))
        importance_ratio = tf.exp(action_log_probs - old_action_log_probs)
        loss_pg = tf.reduce_sum(tf.minimum(
            importance_ratio * advantages,
            tf.clip_by_value(
                importance_ratio,
                1 - epsilon,
                1 + epsilon) * advantages
        )) / batch_size
        loss_vf = mse(tf.squeeze(returns), tf.squeeze(critic_net(obs, mask)))
        total_loss = -loss_pg + c1 * loss_vf
    grads = tape.gradient(total_loss, trainable_variables)
    # record loss values to metrics
    loss_pg_metric(loss_pg)
    loss_vf_metric(loss_vf)
    return grads


# In[ ]:


def train_minibatch(
        obs,
        mask,
        actions,
        action_means,
        old_action_log_probs,
        returns,
        advantages,
        stddev=1e-3,
        epsilon=.2,
        c1=1,
        num_epochs=10,
        minibatch_size=50):
    for i in range(num_epochs):
        minibatch = np.random.choice(obs.shape[0], size=minibatch_size)
        grads = calculate_gradients(
            actor_net, critic_net, trainable_variables,
            tf.gather(obs, indices=minibatch),
            tf.gather(mask, indices=minibatch),
            tf.gather(actions, indices=minibatch),
            tf.gather(action_means, indices=minibatch),
            tf.gather(old_action_log_probs, indices=minibatch),
            tf.gather(returns, indices=minibatch),
            tf.gather(advantages, indices=minibatch),
            stddev=stddev,
            epsilon=epsilon,
            c1=c1)
        optimizer.apply_gradients(zip(grads, trainable_variables))


# Train minibatch, record the loss values and infidelity for the episode, and update layers with new weights.

# In[ ]:


# # increment global step by number of timesteps that are being trained on
# global_step.assign_add(obs.shape[0])
# train_minibatch(
#     obs,
#     mask,
#     actions,
#     action_means,
#     old_action_log_probs,
#     returns,
#     advantages,
#     stddev=1e-3,
#     epsilon=.2,
#     c1=1,
#     num_epochs=10,
#     minibatch_size=50
# )

# with train_summary_writer.as_default():
#     global_step_np = global_step.numpy()
#     tf.summary.scalar('loss_pg', loss_pg_metric.result(), step=global_step_np)
#     tf.summary.scalar('loss_vf', loss_vf_metric.result(), step=global_step_np)
#     tf.summary.scalar('infidelity', infidelity_metric.result(), step=global_step_np)

# loss_pg_metric.reset_states()
# loss_vf_metric.reset_states()
# infidelity_metric.reset_states()
# stateful_actor_net.layers[0].set_weights(actor_net.layers[0].get_weights())


# ## Write a PPO experience collection and training loop

# In[ ]:


def ppo_loop(
        stddev=1e-2,
        epsilon=.2,
        c1=1,
        num_epochs=5,
        minibatch_size=25,
        save_weights=False,
        evaluate=False):
    """Collects experience, trains networks
    """
    (
        obs, mask, actions, action_means,
        rewards, step_types, discounts, returns,
        advantages, old_action_log_probs
    ) = collect_experience(
        num_steps=500,
        stddev=stddev,
        max_sequence_length=100
    )
    print('collected experience')
    global_step.assign_add(obs.shape[0])
    global_step_np = global_step.numpy()
    # train
    train_minibatch(
        obs,
        mask,
        actions,
        action_means,
        old_action_log_probs,
        returns,
        advantages,
        stddev=stddev,
        epsilon=epsilon,
        c1=c1,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size)
    print('trained networks')
    stateful_actor_net.layers[0].set_weights(actor_net.layers[0].get_weights())
    stateful_actor_net.reset_states()
    print('reset state')
    with train_summary_writer.as_default():
        tf.summary.scalar('loss_pg',
                          loss_pg_metric.result(),
                          step=global_step_np)
        tf.summary.scalar('loss_vf',
                          loss_vf_metric.result(),
                          step=global_step_np)
        tf.summary.scalar('infidelity',
                          infidelity_metric.result(),
                          step=global_step_np)
    loss_pg_metric.reset_states()
    loss_vf_metric.reset_states()
    infidelity_metric.reset_states()
    if evaluate:
        # evaluate the actor with noise-free actions
        rewards, control_amplitudes = evaluate_actor()
        np.savez_compressed(
            os.path.join(
                'logs', current_time,
                'controls', f'{global_step.numpy():06.0f}'),
            control_amplitudes=control_amplitudes.numpy()
        )
        with test_summary_writer.as_default():
            tf.summary.scalar('infidelity',
                              infidelity_metric.result(),
                              step=global_step_np)
        infidelity_metric.reset_states()
    print('recorded metrics')
    if save_weights:
        actor_net.save_weights(os.path.join(
            'logs', current_time,
            'models', f'actor-{global_step.numpy():06.0f}'
        ))
        critic_net.save_weights(os.path.join(
            'logs', current_time,
            'models', f'critic-{global_step.numpy():06.0f}'
        ))
        print('saved model weights')


# In[ ]:


for i in range(int(1e3)):
    print(f'iteration {i}')
    save_weights = i % 20 == 0
    evaluate = i % 10 == 0
    ppo_loop(
        stddev=1e-2,
        epsilon=.2,
        c1=1e2,
        num_epochs=10,
        minibatch_size=75,
        save_weights=save_weights,
        evaluate=evaluate
    )
