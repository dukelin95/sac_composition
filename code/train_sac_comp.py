
# import argparse
import tensorflow as tf

from sac.algos import SAC
from sac.misc.utils import timestamp
from sac.misc.instrument import run_sac_experiment
from sac.replay_buffers import SimpleReplayBuffer, DemoReplayBuffer
from sac.misc.sampler import SimpleSampler
from sac.value_functions import NNQFunction, NNVFunction
from sac.policies import UniformPolicy, GaussianPtrPolicy

from rllab.envs.normalized_env import normalize
from robosuite.environments.sawyer_test_reach import SawyerReach
from robosuite.environments.sawyer_test_reachpick import SawyerReachPick
from robosuite.models.tasks import UniformRandomSampler

from crl_env_wrapper import CRLWrapper
from ik_wrapper import IKWrapper

ENVIRONMENTS = {
    'sawyer-reach': {
        'default': SawyerReach
    },
    'sawyer-reach-pick': {
        'default': SawyerReachPick
    }
}
DEFAULT_DOMAIN = DEFAULT_ENV = 'sawyer-reach'
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())


def run_experiment(variant):
    sub_level_policies_paths=[]
    args = arg()

    if args.domain=='sawyer-reach':
        print("Composition Reach")
        goal_size = 0
        sub_level_policies_paths.append("ikx")
        sub_level_policies_paths.append("iky")
        sub_level_policies_paths.append("ikz")
        random_arm_init = [-0.1, 0.1]
        render = False
        reward_shaping = True
        horizon = 250
        env = normalize(
                CRLWrapper(
                    IKWrapper(
                        SawyerReach(
                            # playable params
                            random_arm_init=random_arm_init,
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
        ep_length = 1500

    elif args.domain == 'sawyer-reach-pick':
        print("Composition Reach and Pick")
        goal_size = 3
        sub_level_policies_paths.append("log/prim/pick/2019-08-14-18-18-17-370041-PDT/itr_2000.pkl")
        sub_level_policies_paths.append("log/prim/reach/2019-08-20-15-52-39-191438-PDT/itr_2000.pkl")

        render = False

        random_arm_init = [-0.0001, 0.0001]
        reward_shaping = False
        horizon = 1000
        env = normalize(
            CRLWrapper(
                    SawyerReachPick(
                        # playable params
                        random_arm_init=random_arm_init,
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
        ep_length = 1500

    elif args.domain == 'sawyer-reach-pick-simple':
        print("Composition Reach and Pick Simple")
        goal_size = 3
        sub_level_policies_paths.append("log/prim/pick/2019-08-14-18-18-17-370041-PDT/itr_2000.pkl")
        sub_level_policies_paths.append("log/prim/reach/2019-08-20-15-52-39-191438-PDT/itr_2000.pkl")

        render = False

        random_arm_init = [-0.0001, 0.0001]
        reward_shaping = False
        horizon = 500
        env = normalize(
            CRLWrapper(
                SawyerReachPick(
                    # playable params
                    random_arm_init=random_arm_init,
                    has_renderer=render,
                    reward_shaping=reward_shaping,
                    horizon=horizon,
                    placement_initializer=UniformRandomSampler(
                                    x_range=[-0.01, 0.01],
                                    y_range=[-0.01, 0.01],
                                    ensure_object_boundary_in_range=False,
                                    z_rotation=None,
                                ),
                    # constant params
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    use_object_obs=True,
                    control_freq=100, )
            )
        )
        ep_length = 3000
    else:
        raise ValueError("Domain not available")

    if args.demo:
        pool = DemoReplayBuffer(
            env_spec=env.spec,
            max_replay_buffer_size=1e6,
            seq_len=len(sub_level_policies_paths),
        )
    else:
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
        epoch_length=ep_length,
        n_epochs=5e3,
        # n_epochs=5,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
        sampler=sampler,
        use_demos=args.demo,
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


def launch_experiments(args):

    num_experiments = 1
    print('Launching {} experiments.'.format(num_experiments))
    for i in range(num_experiments):
        print("Experiment: {}/{}".format(i+1, num_experiments))

        run_sac_experiment(
            run_experiment,
            mode='local',
            n_parallel=1,
            terminate_machine=True,
            log_dir='/root/code/log/{0}/{1}'.format(args.domain, timestamp()),

            snapshot_mode='gap',
            snapshot_gap=100,
            sync_s3_pkl=True,
        )

class arg():
    def __init__(self):
        self.domain = 'sawyer-reach-pick'
        self.demo = True


def main():
    args = arg()
    launch_experiments(args)


if __name__ == '__main__':
    main()