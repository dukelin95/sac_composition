
# import argparse
import tensorflow as tf

from sac.algos import SAC
from sac.misc.utils import timestamp
from sac.misc.instrument import run_sac_experiment
from sac.replay_buffers import SimpleReplayBuffer
from sac.misc.sampler import SimpleSampler
from sac.value_functions import NNQFunction, NNVFunction
from sac.policies import UniformPolicy, GaussianPtrPolicy

from rllab.envs.normalized_env import normalize
from robosuite.environments.sawyer_test_reach import SawyerReach

from crl_env_wrapper import CRLWrapper
from ik_wrapper import IKWrapper

ENVIRONMENTS = {
    'sawyer-reach': {
        'default': SawyerReach
    }
}
DEFAULT_DOMAIN = DEFAULT_ENV = 'sawyer-reach'
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--domain',
#                         type=str,
#                         choices=AVAILABLE_DOMAINS,
#                         default='sawyer-reach')
#     parser.add_argument('--policy',
#                         type=str,
#                         choices=('gaussian', 'gaussian_ptr'),
#                         default='gaussian')
#     parser.add_argument('--env', type=str, default=DEFAULT_ENV)
#     parser.add_argument('--exp_name', type=str, default=timestamp())
#     parser.add_argument('--mode', type=str, default='local')
#     parser.add_argument('--log_dir', type=str, default='/root/code/log')
#     args = parser.parse_args()
#
#     return args


def run_experiment(variant):
    sub_level_policies_paths=[]
    # args = parse_args()
    args = arg()

    if args.domain=='sawyer-reach':
        goal_size = 0
        sub_level_policies_paths.append("ikx")
        sub_level_policies_paths.append("iky")
        sub_level_policies_paths.append("ikz")
        random_arm_init = [-0.1, 0.1]
        lower_goal_range = [-0.1, -0.1, -0.1]
        upper_goal_range = [0.1, 0.1, 0.1]
        render = False
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
                            control_freq=100,)
                        )
                    )
                )
    else:
        raise ValueError("Domain not available")

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=1e6,
        seq_len=len(sub_level_policies_paths),
    )
    sampler = SimpleSampler(
        max_path_length=horizon-1, # should be same as horizon
        min_pool_size=1000,
        batch_size=256

    )
    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=2e3,
        # n_epochs=5,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
        sampler=sampler
    )
    M = 128
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)
    policy=GaussianPtrPolicy(env_spec=env.spec,hidden_layer_sizes=(M,M),reparameterize=True,reg=1e-3,)

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        g=goal_size,
        policy=policy,
        sub_level_policies_paths=sub_level_policies_paths,
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

class arg ():
    def __init__(self):
        self.exp_name = 'test'
        self.domain = 'sawyer-reach'
        self.task = 'default'

def launch_experiments():
    args = arg()
    num_experiments = 1
    print('Launching {} experiments.'.format(num_experiments))
    for i in range(num_experiments):
        print("Experiment: {}/{}".format(i+1, num_experiments))
        experiment_prefix = args.domain + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=args.domain, exp_name=args.exp_name, i=0)

        run_sac_experiment(
            run_experiment,
            mode='local',
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            terminate_machine=True,
            log_dir='/root/code/log/{0}/{1}'.format(args.domain, timestamp()),

            snapshot_mode='gap',
            snapshot_gap=100,
            sync_s3_pkl=True,
        )
def main():
    # args = parse_args()
    launch_experiments()


if __name__ == '__main__':
    main()