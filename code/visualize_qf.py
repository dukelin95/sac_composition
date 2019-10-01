from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import robosuite
from robosuite.controllers import SawyerIKController
import robosuite.utils.transform_utils as T

import argparse
import joblib
import os

from robosuite.environments.sawyer_test_reach import SawyerReach
from ik_wrapper import IKWrapper
from rllab.envs.normalized_env import normalize
from crl_env_wrapper import CRLWrapper

import math, random, time

def fibonacci_sphere(length = 1, samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([length*x,length*y,length*z])

    return points


class inverse_kin:

    def __init__(self, env):
        self.env = env.wrapped_env.env
        self.controller = SawyerIKController(
            bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
            robot_jpos_getter=self._robot_jpos_getter,
        )


    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def xyz_to_jvel(self, xyz_action):
        constant_quat = np.array([0, 0, 0, 1])
        input_1 = self._make_input(np.concatenate((xyz_action[:3], constant_quat)), self.env._right_hand_quat)

        velocities = self.controller.get_control(**input_1)
        gripper_action = np.ones(1)
        low_action = np.concatenate([velocities, gripper_action])
        return low_action

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        return {
            "dpos": action[:3],
            # IK controller takes an absolute orientation in robot base frame
            "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
        }

fig = plt.figure()
ax = Axes3D(fig)

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='Path to the snapshot file.')
args = parser.parse_args()

# reset env to get initial obs = initial state
render = False
random_arm_init = [-0.1, 0.1]
reward_shaping = True
horizon = 250
env = normalize(
    CRLWrapper(
        # IKWrapper(
            SawyerReach(
                # playable params
                random_arm_init=random_arm_init,
                seed=True,
                has_renderer=render,
                reward_shaping=reward_shaping,
                horizon=horizon,

                # constant params
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=True,
                control_freq=100, )
        # )
    )
)
obs = env.reset()
ik = inverse_kin(env)
start = obs[0:3]

points = fibonacci_sphere(0.1, 1000)
x_actions = np.array([points[i][0] for i in range(len(points))]) + start[0]
y_actions = np.array([points[i][1] for i in range(len(points))]) + start[1]
z_actions = np.array([points[i][2] for i in range(len(points))]) + start[2]

Do = obs.size
low, high = env.wrapped_env.env.action_spec
Da = low.size

obs_pl = tf.placeholder(
    tf.float32,
    shape=[None, Do],
    name='observation',
)

action_pl = tf.placeholder(
    tf.float32,
    shape=[None, Da],
    name='actions',
)
start_time = time.time()
with tf.Session() as session:
    # load network from pkl file, snapshot['qf1']
    # snapshot = joblib.load('log/prim/reach/2019-08-20-15-52-39-191438-PDT/itr_2000.pkl')
    snapshot = joblib.load(args.file)
    qf1_snap = snapshot['qf1']
    qf1 = qf1_snap.get_output_for(obs_pl, action_pl, reuse=True)
    # discretize xyz somehow
    # pass into IK to get action
    q_vals = []
    for i in range(len(x_actions)):

        xyz_action = np.array([x_actions[i],
                               y_actions[i],
                               z_actions[i]])
        vel_action = ik.xyz_to_jvel(xyz_action)

        # pass into network, get single value?
        feed = {
            obs_pl: obs.reshape(1,-1),
            action_pl: vel_action.reshape(1,-1),
        }
        q_vals.append(session.run(qf1, feed))


# normalize q_vals
q_vals = np.array(q_vals)
normalized_q = (q_vals - np.mean(q_vals))/np.std(q_vals)

print(time.time() - start_time)
# graph
goal = env.wrapped_env.env.goal

# x_actions.append(goal[0])
# y_actions.append(goal[1])
# z_actions.append(goal[2])

print("Start: {0}, Goal: {1}".format(start, goal))
plot = ax.scatter(x_actions, y_actions, z_actions, c=normalized_q)
ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]])
fig.colorbar(plot)

plt.show()