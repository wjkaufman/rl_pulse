#!/usr/bin/env python
# coding: utf-8

# # Pulse design using reinforcement learning
# _Written by Will Kaufman, November 2020_
#
# adapted from the jupyter notebook


import numpy as np
import os
# import time
import qutip as qt
import tensorflow as tf
import datetime

from rl_pulse.environments import spin_system_continuous


# ## Define algorithm hyperparameters

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
X = qt.tensor([qt.sigmax()]*N)
Y = qt.tensor([qt.sigmay()]*N)
# Z = qt.tensor([qt.sigmaz()]*N)
Hcontrols = [50e3 * X, 50e3 * Y]
target = qt.propagator(X, np.pi/4)

# define environment
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
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# count number of time steps (or episodes?) played through
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

def calculate_returns(rewards, step_types, gamma=0.99):
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
        returns[-(i + 1)] = (gamma
                             * returns[-i]
                             * tf.cast(step_types[-(i + 1)] == 1,
                                       tf.float32)) + rewards[-(i+1)]
    return returns


def get_obs_and_mask(obs, max_sequence_length=500):
    obs2 = []
    mask = []
    num_features = obs[0].shape[-1]
    for i in range(len(obs)):
        obs_length = obs[i].shape[-2]
        obs2.append(tf.concat(
            [tf.cast(obs[i], tf.float32), tf.zeros((1, max_sequence_length - obs_length, num_features))],
            axis=1
        ))
        mask.append(tf.concat(
            [tf.ones((1, obs_length)), tf.zeros((1, max_sequence_length - obs_length))],
            axis=1
        ))
    obs2 = tf.squeeze(tf.stack(obs2))
    mask = tf.squeeze(tf.stack(mask))
    return obs2, mask


def collect_data(num_steps=100, stddev=1e-3, max_sequence_length=100):
    step_types = []
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
        action_mean = stateful_actor_net(tf.expand_dims(step.observation[:, -1, :], 1)) #collect_action(step.observation, stddev=stddev)
        action = action_mean + tf.random.normal(shape=action_mean.shape, stddev=stddev)
        actions.append(action)
        action_means.append(action_mean)
        step = env.step(action)
        rewards.append(step.reward)
        step_types.append(step.step_type)
        if step.step_type == 2:
            stateful_actor_net.reset_states()
            step = env.reset()
    # put data into tensors
    step_types = tf.stack(step_types)
    actions = tf.squeeze(tf.stack(actions))
    action_means = tf.squeeze(tf.stack(action_means))
    rewards = tf.stack(rewards)
    # reshape observations to be same sequence length, and create
    # a mask for original sequence length
    obs, mask = get_obs_and_mask(observations, max_sequence_length)
    returns = tf.stack(calculate_returns(rewards, step_types))
    advantages = returns - critic_net(obs, mask=mask)
    old_action_log_probs = tf.reduce_sum(-(actions - action_means)**2 / stddev**2, axis=1, keepdims=True)
    return (obs, mask, actions, action_means,
            rewards, step_types, returns,
            advantages, old_action_log_probs)


# %lprun -f collect_data collect_data()
(obs, mask, actions,
 action_means, rewards, step_types, returns,
 advantages, old_action_log_probs) = collect_data(stddev=stddev)


# ## Training
#
# To "train" the actor and critic networks (change the network parameters to minimize the loss function), an optimizer using the Adam algorithm is used. The loss function is described above, and is composed of policy-gradient loss and value function loss.

optimizer = tf.optimizers.Adam()
mse = tf.losses.mse


if not actor_net.built:
    actor_net.build(input_shape=(None, None, 2))


# Define a list of trainable variables that should be updated when minimizing the loss function.

critic_vars = critic_net.trainable_variables
actor_vars = actor_net.trainable_variables
trainable_variables = set()
for var in critic_vars + actor_vars:
    trainable_variables.add(var.ref())
trainable_variables = list(trainable_variables)
trainable_variables = [var.deref() for var in trainable_variables]


def grad(
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
        action_log_probs = tf.reduce_sum(-(actions - actor_net(obs, mask))**2 / stddev**2,
                                         axis=1,
                                         keepdims=True)
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
    return grads, pg_loss, vf_loss


def train_minibatch(obs, mask, actions, action_means, old_action_log_probs, returns, advantages,
                    actor_net, critic_net, trainable_variables, stddev=1e-3,
                    num_epochs=10, minibatch_size=100, printing=False):
    for i in range(num_epochs):
        minibatch = np.random.choice(obs.shape[0], size=minibatch_size)
        grads, loss_clip, loss_value = grad(
            actor_net, critic_net, trainable_variables,
            tf.gather(obs, indices=minibatch),
            tf.gather(mask, indices=minibatch),
            tf.gather(actions, indices=minibatch),
            tf.gather(action_means, indices=minibatch),
            tf.gather(old_action_log_probs, indices=minibatch),
            tf.gather(returns, indices=minibatch),
            tf.gather(advantages, indices=minibatch),
            stddev=stddev)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        if printing:
            print(f'{i}:\tloss_clip: {loss_clip}\tloss_value: {loss_value}')


# increment global step by number of timesteps that are being trained on
global_step.assign_add(obs.shape[0])


train_minibatch(obs, mask, actions, action_means, old_action_log_probs, returns, advantages,
                actor_net, critic_net, trainable_variables, stddev=stddev,
                num_epochs=10, minibatch_size=50)


tf.reduce_sum(rewards)


env.fidelity()


with train_summary_writer.as_default():
    step = global_step.numpy()
    tf.summary.scalar('pg_loss', pg_loss_metric.result(), step=step)
    tf.summary.scalar('vf_loss', vf_loss_metric.result(), step=step)
    tf.summary.scalar('reward', tf.reduce_sum(rewards), step=step)
    tf.summary.scalar('infidelity', 1 - env.fidelity(), step=step)

pg_loss_metric.reset_states()
vf_loss_metric.reset_states()


stateful_actor_net.layers[0].set_weights(actor_net.layers[0].get_weights())
stateful_actor_net.reset_states()


# ## Write a training loop

def train_step(save_weights=True):
    """Collect experience, train networks
    """
    (obs, mask, actions,
     action_means, rewards, step_types, returns,
     advantages, old_action_log_probs) = collect_data()
    print('collected data')
    global_step.assign_add(obs.shape[0])
    # train
    train_minibatch(obs, mask, actions, action_means,
                    old_action_log_probs, returns, advantages,
                    actor_net, critic_net, trainable_variables,
                    num_epochs=5, minibatch_size=64)
    print('trained networks')
    with train_summary_writer.as_default():
        step = global_step.numpy()
        tf.summary.scalar('pg_loss', pg_loss_metric.result(), step=step)
        tf.summary.scalar('vf_loss', vf_loss_metric.result(), step=step)
        tf.summary.scalar('reward', tf.reduce_sum(rewards), step=step)
        tf.summary.scalar('infidelity', 1 - env.fidelity(), step=step)
    pg_loss_metric.reset_states()
    vf_loss_metric.reset_states()
    print('recorded metrics')
    stateful_actor_net.layers[0].set_weights(actor_net.layers[0].get_weights())
    stateful_actor_net.reset_states()
    print('reset state')
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


for i in range(int(1e4)):
    train_step(save_weights=(i % 200 == 0))
