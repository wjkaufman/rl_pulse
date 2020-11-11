#!/usr/bin/env python
# coding: utf-8
# # Pulse design using reinforcement learning
# _Written by Will Kaufman, October 2020_

import numpy as np
import os
import qutip as qt
import tensorflow as tf
import datetime
from rl_pulse.environments import spin_system_continuous

# import importlib
# importlib.reload(spin_system_continuous)

# ## Define algorithm hyperparameters
#
#

# TODO eventually fill in hyperparameters at top of doc
discount = 0.99
stddev = 1e-3

# ## Initialize the spin system

N = 3  # 4-spin system

chemical_shifts = np.random.normal(scale=50, size=(N,))
Hcs = sum(
    [qt.tensor(
        [qt.identity(2)]*i
        + [chemical_shifts[i] * qt.sigmaz()]
        + [qt.identity(2)]*(N-i-1)
    ) for i in range(N)]
)

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

env = spin_system_continuous.SpinSystemContinuousEnv(
    Hsys=Hsys,
    Hcontrols=Hcontrols,
    target=target,
    discount=discount
)

# ## Define metrics

pg_loss_metric = tf.keras.metrics.Mean('pg_loss', dtype=tf.float32)
vf_loss_metric = tf.keras.metrics.Mean('vf_loss', dtype=tf.float32)

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

# count number of time steps played through
global_step = tf.Variable(0, trainable=False, name='global_step')

# ## Define actor and critic networks

lstm = tf.keras.layers.LSTM(64)
stateful_lstm = tf.keras.layers.LSTM(64, stateful=True)
hidden1 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
hidden2 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
policy = tf.keras.layers.Dense(2, activation=tf.keras.activations.tanh)
value = tf.keras.layers.Dense(1)

actor_net = tf.keras.models.Sequential([
    lstm,
    hidden1,
    hidden2,
    policy
])

critic_net = tf.keras.models.Sequential([
    lstm,
    hidden1,
    hidden2,
    value
])

stateful_actor_net = tf.keras.models.Sequential([
    stateful_lstm,
    hidden1,
    hidden2,
    policy
])

# ## Define PPO agent


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

# %lprun -f collect_experience collect_experience()

# (
#     obs, mask, actions, action_means,
#     rewards, step_types, discounts, returns,
#     advantages, old_action_log_probs
# ) = collect_experience(num_steps=100,
#                        stddev=stddev,
#                        max_sequence_length=100)

# ## Evaluate the actor


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
    return rewards, tf.squeeze(step.observation)

# a, b = evaluate_actor()

# plt.plot(a.numpy())

# plt.plot(b.numpy()[:,0], label='x')
# plt.plot(b.numpy()[:,1], label='y')
# plt.legend()

# ## Training
#


optimizer = tf.optimizers.Adam()
mse = tf.losses.mse

if not critic_net.built:
    critic_net.build(input_shape=(None, None, 2))
if not actor_net.built:
    actor_net.build(input_shape=(None, None, 2))


critic_vars = critic_net.trainable_variables
actor_vars = actor_net.trainable_variables
trainable_variables = set()
for var in critic_vars + actor_vars:
    trainable_variables.add(var.ref())
trainable_variables = list(trainable_variables)
trainable_variables = [var.deref() for var in trainable_variables]


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
        pg_loss = tf.reduce_sum(tf.minimum(
            importance_ratio * advantages,
            tf.clip_by_value(
                importance_ratio,
                1 - epsilon,
                1 + epsilon) * advantages
        )) / batch_size
        vf_loss = mse(tf.squeeze(returns), tf.squeeze(critic_net(obs, mask)))
        total_loss = -pg_loss + c1 * vf_loss
    grads = tape.gradient(total_loss, trainable_variables)
    # record loss values to metrics
    pg_loss_metric(pg_loss)
    vf_loss_metric(vf_loss)
    return grads


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


# ## Write a PPO experience collection and training loop

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
    ) = collect_experience()
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
        tf.summary.scalar('pg_loss',
                          pg_loss_metric.result(),
                          step=global_step_np)
        tf.summary.scalar('vf_loss',
                          vf_loss_metric.result(),
                          step=global_step_np)
        tf.summary.scalar('reward',
                          tf.reduce_sum(rewards),
                          step=global_step_np)
        tf.summary.scalar('infidelity',
                          1 - env.fidelity(),
                          step=global_step_np)
    pg_loss_metric.reset_states()
    vf_loss_metric.reset_states()
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
            tf.summary.scalar('reward',
                              tf.reduce_sum(rewards),
                              step=global_step_np)
            tf.summary.scalar('infidelity',
                              1 - env.fidelity(),
                              step=global_step_np)
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


for i in range(1e3):
    print(f'iteration {i}')
    save_weights = i % 200 == 0
    evaluate = i % 50 == 0
    ppo_loop(
        stddev=1e-2,
        epsilon=.3,
        c1=1e2,
        num_epochs=10,
        minibatch_size=25,
        save_weights=save_weights,
        evaluate=evaluate
    )
