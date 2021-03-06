import time
import robosuite as suite
# from robosuite.wrappers.gym_wrapper import GymWrapper
#
# from gym_goal_wrapper import GymGoalEnvWrapper
# from ik_wrapper import IKWrapper
import numpy as np

# from stable_baselines.ddpg.policies import MlpPolicy
# from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
# from stable_baselines import DDPG
# from stable_baselines import HER
#
# from utils import HERGoalEnvWrapper
# # from sawyer_primitive_reach import SawyerPrimitiveReach
# from test_sawyer_xyz import SawyerPrimitiveReach
# from test_sawyer_pick import SawyerPrimitivePick
# import argparse
from robosuite.environments.sawyer_test_pick import SawyerPrimitivePick
from robosuite.environments.sawyer_test_reach import SawyerReach
from robosuite.environments.sawyer_test_reachpick import SawyerReachPick

from crl_env_wrapper import CRLWrapper
from ik_wrapper import IKWrapper
from rllab.envs.normalized_env import normalize
from robosuite.models.tasks import UniformRandomSampler
from crl_4DOF_env_wrapper import CRL4DOFWrapper
render = True

random_arm_init = [-0.0001, 0.0001]
reward_shaping = False
horizon = 500
# env = normalize(
from robosuite.environments.sawyer_test_pick import SawyerPrimitivePick
random_arm_init = [-0.0, 0.0]
# random_arm_init = None
instructive = 1.0
decay = 3e-6
env = SawyerPrimitivePick(
    instructive=instructive,
    decay=decay,
    random_arm_init=random_arm_init,
    has_renderer=True,
    reward_shaping=False,
    horizon=150,
    ignore_done =True,
    placement_initializer=UniformRandomSampler(
        x_range=[-0.01, 0.01],
        y_range=[-0.01, 0.01],
        ensure_object_boundary_in_range=False,
        z_rotation=None,
    ),

    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=100, )

env.reset()
env.viewer.set_camera(camera_id=0)
env.render()


def cube_pos(env):
   print(np.array(env.env.env.sim.data.body_xpos[env.env.env.cube_body_id]))
def eef_pos(env):
   print(np.array(env.env.env.sim.data.site_xpos[env.env.env.eef_site_id]))

def joint_pos(env):
   print(np.array([env.env.env.sim.data.qpos[x] for x in env.env.env._ref_joint_pos_indexes]))

def test_ik():
   import pybullet as p
   import os
   from os.path import join as pjoin
   import robosuite

   constant_quat = np.array([-0.01704371, -0.99972409, 0.00199679, -0.01603944])
   target_position = np.array([0.58038172, -0.01562932, 0.90211762]) \
                              + np.random.uniform(-0.02, 0.02, 3)
   bullet_path = os.path.join(robosuite.models.assets_root, "bullet_data")
   robot_urdf = pjoin(bullet_path, "sawyer_description/urdf/sawyer_arm.urdf")
   ik_robot = p.loadURDF(robot_urdf, (0, 0, 0.9), useFixedBase=1)
   soln = list(p.calculateInverseKinematics(
                    ik_robot,
                    6,
                    target_position,
                    targetOrientation=constant_quat,
                    restPoses=[0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161],
                    jointDamping=[0.1] * 7,
                ))
   return soln

def init(env, loop):
   for i in range(loop):
      o = env.reset()
      env.wrapped_env.env.viewer.viewer.add_marker(pos=env.wrapped_env.env.goal,
                                                   size=np.array((0.02, 0.02, 0.02)), label='goal',
                                                   rgba=[1, 0, 0, 0.5])
      env.render()
      print(env.wrapped_env.env.goal)


def find(env1, env3, loop):
    p = []
    q = []
    for i in range(loop):
        o = env3.reset()
        env3.render()
        p.append(np.array(env1.sim.data.site_xpos[env1.eef_site_id]))
        q.append(suite.utils.transform_utils.convert_quat(
                 env1.sim.data.get_body_xquat("right_hand"), to='xyzw'))
    return p, q

def view(env, loop, action=None):
    for i in range(loop):
        #if True:
        if action is None:
           action = np.array([0.0, 0.0, 0.0, np.random.uniform(-1.1), np.random.uniform(-1.1), np.random.uniform(-1.1), np.random.uniform(-1.1), 0])
#           action[np.random.randint(3)] = 0.01
        obs_dict, r, d, i = env.step(action)
        # env.viewer.viewer.add_marker(pos=env.goal, size=np.array((0.02,0.02,0.02)), label='goal', rgba=[1, 0, 0, 0.5])
        # print(action)
        env.render()
