import argparse

import joblib
import numpy as np
import tensorflow as tf

from robosuite.environments.sawyer_test_pick import SawyerPrimitivePick
from robosuite.environments.sawyer_test_reach import SawyerReach
from ik_wrapper import IKWrapper
from rllab.envs.normalized_env import normalize
from crl_env_wrapper import CRLWrapper
from crl_4DOF_env_wrapper import CRL4DOFWrapper
from rllab.misc import tensor_utils

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    j_positions = []
    r_quats = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)

        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        j_positions.append(env.wrapped_env.env._joint_positions)
        r_quats.append(env.wrapped_env.env._right_hand_quat)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.wrapped_env.viewer.set_camera(2)
            env.render()
            env.wrapped_env.env.viewer.viewer.add_marker(pos=env.wrapped_env.env.goal,
                                                             size=np.array((0.01, 0.01, 0.01)), label='goal',
                                                             rgba=[1, 0, 0, 0.5])
            # timestep = 0.05
            # time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        j_positions=tensor_utils.stack_tensor_list(j_positions),
        r_quats=tensor_utils.stack_tensor_list(r_quats),
    )

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

        render = False
        # decay = 5e-4
        # instructive = 0.0
        #
        # random_arm_init = [-0.05, 0.05]
        # reward_shaping = False
        # horizon = 250
        #
        # env = normalize(
        #     CRL4DOFWrapper(
        #         # IKWrapper(
        #         SawyerPrimitivePick(
        #             instructive=instructive,
        #             decay=decay,
        #             random_arm_init=random_arm_init,
        #             has_renderer=render,
        #             reward_shaping=reward_shaping,
        #             horizon=horizon,
        #             has_offscreen_renderer=False,
        #             use_camera_obs=False,
        #             use_object_obs=True,
        #             control_freq=100, )
        #         , use_gripper=True)
        # )
        # )
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
                    control_freq=100, )
                # )
                , use_gripper=True)
        )

        mean_reward = []
        success = 0

        with policy.deterministic(args.deterministic):
            for _ in range(args.trials):
                path = rollout(env,
                               policy,
                               max_path_length=horizon - 1,
                               animated=render,
                               speedup=1.5,
                               always_return_paths=True)
                print(np.sum(path["rewards"]))
                if np.sum(path["rewards"])> -0.0: success += 1
                mean_reward.append(np.sum(path["rewards"]))
        print("Average Return {}+/-{}".format(np.mean(mean_reward),
                                              np.std(mean_reward)))
        print("Success {}/{}".format(success,args.trials))

        return path


if __name__ == "__main__":
    args = parse_args()
    res = simulate_policy(args)
