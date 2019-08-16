import argparse

import numpy as np
import tensorflow as tf
import joblib
import os.path as osp


from rllab.envs.normalized_env import normalize
from rllab.misc import tensor_utils

from sac.misc import tf_utils
from sac.policies.ik_policy import IK_Policy
from robosuite.environments.sawyer_test_reach import SawyerReach

from crl_env_wrapper import CRLWrapper
from ik_wrapper import IKWrapper
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    # parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    # parser.add_argument('--speedup', '-s', type=float, default=10)
    # parser.add_argument('--domain',type=str,default='ant-cross-maze')
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    # parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
    #                         action='store_false')
    # parser.add_argument('--log_dir', type=str, default=None, dest='FilePath')
    parser.add_argument('--trials', type=int, default=5, dest='trials')
    # parser.add_argument('--min_reward', type=int, default=-3000, dest='reward_min')
    # parser.add_argument('--policy_h', type=int)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    return args

def rollout(env, policy,sub_level_policies,path_length=1000, render=True, speedup=10, g=2):
    running_paths = dict(
                observations=[],
                actions=[],
                rewards=[],
                env_infos=[],
                agent_infos=[],
                goal=[],
        )
    observation = env.reset()
    policy.reset()
    t = 0
    obs = observation
    for t in range(path_length):


        sub_level_actions=[]
        if g!=0:
            obs=observation[:-g]
        else:
            obs=observation
        for i in range(0,len(sub_level_policies)):
            action, _ = sub_level_policies[i].get_action(obs)
            sub_level_actions.append(action.reshape(1,-1))
        sub_level_actions=np.stack(sub_level_actions,axis=0)
        sub_level_actions=np.transpose(sub_level_actions,(1,0,2))

        action, agent_info = policy.get_action(observation,sub_level_actions)
        print(action)
        next_obs, reward, terminal, env_info = env.step(action)

        running_paths["observations"].append(observation)
        running_paths["actions"].append(action)
        running_paths["rewards"].append(reward)
        running_paths["env_infos"].append(env_info)
        running_paths["agent_infos"].append(agent_info)
        running_paths["goal"].append(observation[-2:])

        observation = next_obs

        if render:
                env.render()
                env.wrapped_env.env.env.viewer.viewer.add_marker(pos=env.wrapped_env.env.env.goal, size=np.array((0.02, 0.02, 0.02)), label='goal',
                                             rgba=[1, 0, 0, 0.5])
                # time_step = 0.05
                # time.sleep(time_step / speedup)

        if terminal:
           return running_paths

    return running_paths


def simulate_policy(args):
    paths = []
    sub_level_policies=[]
    sub_level_paths = ['ikx', 'iky', 'ikz']
    with tf.Session() as sess:
        for p in range(0, len(sub_level_paths)):
            path = sub_level_paths[p]
            if path[:2] == 'ik':
                with tf.variable_scope(str(p), reuse=False):
                    policy_snapshot = IK_Policy(path[2])
                sub_level_policies.append(policy_snapshot)
            else:
                with tf.variable_scope(str(p), reuse=False):
                    policy_snapshot = joblib.load(sub_level_paths[p])
                sub_level_policies.append(policy_snapshot["policy"])

        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            # env = data['algo'].env
        else:
            policy = data['policy']
            # env = data['env']

        random_arm_init = [-0.1, 0.1]
        lower_goal_range = [-0.1, -0.1, -0.1]
        upper_goal_range = [0.1, 0.1, 0.1]
        render = True
        reward_shaping = True
        horizon = 250
        env = normalize(
            CRLWrapper(
                IKWrapper(
                    SawyerReach(
                        # playable params
                        random_arm_init=random_arm_init,
                        lower_goal_range=lower_goal_range,
                        upper_goal_range=upper_goal_range,
                        has_renderer=render,
                        reward_shaping=reward_shaping,
                        horizon=horizon,

                        # constant params
                        has_offscreen_renderer=False,
                        use_camera_obs=False,
                        use_object_obs=True,
                        control_freq=100, )
                )
            )
        )
        
        mean_reward = []
        with policy.deterministic(args.deterministic):
            Npath = 0
            while Npath<args.trials:
                path = rollout(env, policy,sub_level_policies,path_length=horizon-1,g=0)
                # if np.sum(path["rewards"])>=args.reward_min:
                #     print(np.sum(path["rewards"]))
                #     paths.append(dict(
                #         observations= env.observation_space.flatten_n(path["observations"]),
                #         actions= env.observation_space.flatten_n(path["actions"]),
                #         rewards= tensor_utils.stack_tensor_list(path["rewards"]),
                #         env_infos= path["env_infos"],
                #         agent_infos= path["agent_infos"],
                #         goal = path["goal"]
                #     ))
                Npath+=1
                mean_reward.append(np.sum(path["rewards"]))
            print("Average Return {}+/-{}".format(np.mean(mean_reward),
                                                      np.std(mean_reward)))
            # fileName = osp.join(args.FilePath,'itr.pkl')
            # joblib.dump(paths,fileName, compress=3)
        return env
if __name__ == "__main__":
    args = parse_args()
    e = simulate_policy(args)