import argparse

import joblib
import numpy as np
import tensorflow as tf

from rllab.sampler.utils import rollout
from robosuite.environments.sawyer_test_pick import SawyerPrimitivePick
from robosuite.environments.sawyer_test_reach import SawyerReach
from ik_wrapper import IKWrapper
from rllab.envs.normalized_env import normalize
from crl_env_wrapper import CRLWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    # parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    # parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic',
                        '-d',
                        dest='deterministic',
                        action='store_true')
    parser.add_argument('--trials', type=int, default=5, dest='trials')
    # parser.add_argument('--no-deterministic',
    #                     '-nd',
    #                     dest='deterministic',
    #                     action='store_false')
    # parser.add_argument('--policy_h', type=int)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    return args

def simulate_policy(args):
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            # env = data['algo'].env
        else:
            policy = data['policy']
            # env = data['env']

        instructive = False
        render = True
        decay = 0.3e-6
        random_arm_init = [-0.1, 0.1]
        reward_shaping = False
        horizon = 150

        env = normalize(
            CRLWrapper(
                IKWrapper(
                    SawyerPrimitivePick(
                        instructive=instructive,
                        decay=decay,
                        random_arm_init=random_arm_init,
                        has_renderer=render,
                        reward_shaping=reward_shaping,
                        horizon=horizon,

                        has_offscreen_renderer=False,
                        use_camera_obs=False,
                        use_object_obs=True,
                        control_freq=100, )
                    , gripper=True)
            )
        )

        mean_reward = []
        # while True:
        with policy.deterministic(args.deterministic):
            for _ in range(args.trials):
                path = rollout(env,
                               policy,
                               max_path_length=horizon - 1,
                               animated=True,
                               speedup=1.5,
                               always_return_paths=True)

                mean_reward.append(np.sum(path["rewards"]))
        print("Average Return {}+/-{}".format(np.mean(mean_reward),
                                              np.std(mean_reward)))


if __name__ == "__main__":
    args = parse_args()
    simulate_policy(args)
