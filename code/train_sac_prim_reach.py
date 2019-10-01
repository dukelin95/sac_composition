import numpy as np
import tensorflow as tf

from sac.algos import SAC
from sac.misc.utils import timestamp, unflatten
from sac.policies import GaussianPolicy, LatentSpacePolicy, GMMPolicy, UniformPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.instrument import run_sac_experiment

from robosuite.environments.sawyer_test_reach import SawyerReach
from ik_wrapper import IKWrapper
from rllab.envs.normalized_env import normalize
from crl_env_wrapper import CRLWrapper
from crl_4DOF_env_wrapper import CRL4DOFWrapper

import argparse


def run_experiment(param):
    render = False
    random_arm_init = [-0.0001, 0.0001]
    reward_shaping = False
    horizon = 250
    env = normalize(
        CRL4DOFWrapper(
            # IKWrapper(
                SawyerReach(
                    # playable params
                    random_arm_init=random_arm_init,
                    has_renderer=render,
                    reward_shaping=reward_shaping,
                    horizon=horizon,
                    lower_goal_range=[-0.05, -0.05, -0.05],
                    upper_goal_range=[0.05, 0.05, 0.075],

                    # constant params
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    use_object_obs=True,
                    control_freq=100,)
            # )
        , use_gripper=True)
    )
    replay_buffer_params = {
        'max_replay_buffer_size': 1e6,
    }

    sampler_params = {
        'max_path_length': horizon - 1,
        'min_pool_size': 1000,
        'batch_size': 256,
    }


    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    sampler = SimpleSampler(**sampler_params)

    base_kwargs = dict(
        {
            'epoch_length': 1500,
            'n_train_repeat': 1,
            'n_initial_exploration_steps': 5000,
            'eval_render': False,
            'eval_n_episodes': 1,
            'eval_deterministic': True,
            'n_epochs': 4e3
        },
        sampler=sampler)

    M = 64
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)

    policy = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(64, 64),
        reparameterize=True,
        reg=1e-3,
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        lr=3e-4,
        scale_reward=5,
        discount=0.99,
        tau=0.005,
        reparameterize=True,
        target_update_interval=1,
        action_prior='uniform',
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    algorithm.train()


if __name__ == "__main__":
    run_sac_experiment(
        run_experiment,
        mode='local',
        log_dir='/root/code/log/prim/reach/{0}'.format(timestamp()),
        snapshot_mode='gap',
        snapshot_gap=100,
    )
