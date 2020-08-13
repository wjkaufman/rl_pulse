#!/usr/bin/env python
# coding: utf-8
#
# # Pulse Sequence Design using PPO
# _Written by Will Kaufman, 2020_
# largely inspired by PPO HalfCheetah example
# https://github.com/tensorflow/agents/blob/v0.5.0/tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py

r"""Train and Eval PPO.
To run:
```bash
tensorboard --logdir $HOME/projects/rl_pulse/data/ --port 2223 &
python rl_pulse/tf_agents_propagators_PPO.py
tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

import os
import spin_simulation as ss
import time
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.policies import policy_saver  # , random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.trajectories import trajectory
# from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from environments import spin_sys_discrete


# Define algorithm hyperparameters

def train_eval(
        root_dir,
        episode_length=5,
        # collect parameters
        num_environment_steps=1000000,
        collect_episodes_per_iteration=20,
        num_parallel_environments=20,
        replay_buffer_max_length=1000,
        # training parameters
        num_epochs=25,
        learning_rate=1e-3,
        # evaluation parameters
        num_eval_episodes=5,
        eval_interval=200,
        # summaries and logging parameters
        train_checkpoint_interval=500,
        policy_checkpoint_interval=10000,
        log_interval=1000,
        summaries_flush_secs=1,
        use_tf_functions=True,
        debug_summaries=False,
        summarize_grads_and_vars=False
        ):
    if root_dir is None:
        raise AttributeError('train_eval requires a root_dir.')
    
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    N = 4
    dim = 2**N
    coupling = 1e3
    delta = 500
    (X, Y, Z) = ss.get_total_spin(N=N, dim=dim)
    H_target = ss.get_H_WHH_0(X, Y, Z, delta)

    env = spin_sys_discrete.SpinSystemDiscreteEnv(
        N=4,
        dim=16,
        coupling=coupling,
        delta=delta,
        H_target=H_target,
        X=X, Y=Y,
        delay=5e-6,
        pulse_width=0,
        delay_after=True,
        episode_length=episode_length)

    print('Observation Spec:')
    print(env.time_step_spec().observation)

    print('Reward Spec:')
    print(env.time_step_spec().reward)

    print('Action Spec:')
    print(env.action_spec())

    train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: env] * num_parallel_environments))
    eval_env = tf_py_environment.TFPyEnvironment(env)

    # Define actor and value networks

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=[(32, 3, 1), (32, 3, 1)],
        fc_layer_params=(32, 32),
        # activation_fn=tf.keras.activations.tanh,
        )

    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        conv_layer_params=[(32, 3, 1), (32, 3, 1)],
        fc_layer_params=(32, 32),
        # activation_fn=tf.keras.activations.tanh,
        )

    # Create agent

    # is there a v2 optimizer I could use?
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)

    agent = ppo_clip_agent.PPOClipAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    # random_policy=random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
    #                                                 train_env.action_spec())

    train_env.time_step_spec()

    # Metrics for training/evaluation

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(
            batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=num_parallel_environments),
    ]

    def compute_avg_return(
            environment,
            policy,
            num_episodes=10,
            print_actions=False):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            policy_state = policy.get_initial_state(environment.batch_size)
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step,
                                            policy_state=policy_state)
                policy_state = action_step.state
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                if print_actions:
                    print(f"action: {action_step.action}, "
                          + f"reward: {time_step.reward}, "
                          + f"return: {episode_return}")
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # Create the replay buffer

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length,
    )

    # Add checkpoints and policy saver

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    saved_model = policy_saver.PolicySaver(
        eval_policy, train_step=global_step)

    # Create the driver

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    def train_step():
        trajectories = replay_buffer.gather_all()
        return agent.train(experience=trajectories)

    # Convert functions to `tf_function`s for speedup.
    collect_driver.run = common.function(collect_driver.run, autograph=False)
    agent.train = common.function(agent.train, autograph=False)
    train_step = common.function(train_step)
    # agent.collect_policy.action =common.function(agent.collect_policy.action)

    # Train the agent

    # Reset the train step
    agent.train_step_counter.assign(0)
    global_step.assign(0)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < num_environment_steps:
        global_step_val = global_step.numpy()
        if global_step_val % eval_interval == 0:
            metric_utils.eager_compute(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
            )
        
        start_time = time.time()
        collect_driver.run()
        collect_time += time.time() - start_time

        start_time = time.time()
        total_loss, _ = train_step()
        replay_buffer.clear()
        train_time += time.time() - start_time
        
        for train_metric in train_metrics:
            train_metric.tf_summaries(
                train_step=global_step, step_metrics=step_metrics)

        if global_step_val % log_interval == 0:
            print(f'step = {global_step_val}, loss = {total_loss}')
            steps_per_sec = (
                (global_step_val - timed_at_step)
                / (collect_time + train_time))
            print(f'{steps_per_sec : .3f} steps/sec', steps_per_sec)
            print(f'collect_time = {collect_time:.3f},'
                  + f'train_time = {train_time:.3f}')
        if global_step_val % train_checkpoint_interval == 0:
            train_checkpointer.save(global_step=global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
            policy_checkpointer.save(global_step=global_step_val)
            saved_model_path = os.path.join(
                saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
            saved_model.save(saved_model_path)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

    # Evaluate the agent

    compute_avg_return(eval_env,
                       agent.policy,
                       num_episodes=1,
                       print_actions=True)

    time_step = eval_env.reset()
    print(value_net(time_step.observation,
                    step_type=time_step.step_type)[0].numpy())
    time_step = eval_env.step(1)
    print(value_net(time_step.observation,
                    step_type=time_step.step_type)[0].numpy())
    time_step = eval_env.step(2)
    print(value_net(time_step.observation,
                    step_type=time_step.step_type)[0].numpy())
    time_step = eval_env.step(4)
    print(value_net(time_step.observation,
                    step_type=time_step.step_type)[0].numpy())
    time_step = eval_env.step(3)
    print(value_net(time_step.observation,
                    step_type=time_step.step_type)[0].numpy())
    time_step = eval_env.step(0)
    print(value_net(time_step.observation,
                    step_type=time_step.step_type)[0].numpy())
    print(time_step.reward.numpy())

    # time_step = eval_env.reset()
    # print(actor_net(time_step.observation,
    #                 step_type=time_step.step_type, network_state=()))
                
                
def main():
    # set logging
    
    train_eval(
        os.path.join(os.getcwd(), '..', 'data')
    )  # include hyperparameters here?


if __name__ == '__main__':
    
    # app.run(main)
    main()
