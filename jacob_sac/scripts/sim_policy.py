import argparse

import joblib
import numpy as np
import tensorflow as tf

from rllab.sampler.utils import rollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic',
                        '-d',
                        dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic',
                        '-nd',
                        dest='deterministic',
                        action='store_false')
    parser.add_argument('--policy_h', type=int)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    return args


def simulate_policy(args):
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']

        # with policy.deterministic(args.deterministic):
        mean_reward = []
        # while True:
        for _ in range(10):
            path = rollout(env,
                           policy,
                           max_path_length=args.max_path_length,
                           animated=True,
                           speedup=args.speedup,
                           always_return_paths=True)

            mean_reward.append(np.sum(path["rewards"]))
        print("Average Return {}+/-{}".format(np.mean(mean_reward),
                                              np.std(mean_reward)))


if __name__ == "__main__":
    args = parse_args()
    simulate_policy(args)
