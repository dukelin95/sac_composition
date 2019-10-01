import argparse, pickle, os
import math, random, time
import numpy as np
import tensorflow as tf
import joblib
from tqdm import trange

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import SawyerIKController

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUMPTS = 500

class ik_helper:
    def __init__(self, jpos, quat):
        self.jpos = jpos
        self.quat = quat

        self.controller = SawyerIKController(
            bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
            robot_jpos_getter=self._robot_jpos_getter,
        )


    def _robot_jpos_getter(self):
        return self.jpos


    def xyz_to_jvel(self, xyz_action, jpos, quat):
        self.jpos = jpos
        self.quat = quat

        constant_quat = np.array([0, 0, 0, 1])
        input_1 = self._make_input(np.concatenate((xyz_action[:3], constant_quat)), self.quat)

        self.controller.sync_ik_robot(self._robot_jpos_getter(), simulate=True)
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

    return np.array(points)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the trajectory file.')

    args = parser.parse_args()
    return args


def get_qvals(file, Do, Da, data):
    # setting up qval reading
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
    res = []

    with tf.Session() as session:
        snapshot = joblib.load(file)
        qf1_snap = snapshot['qf1']
        qf1 = qf1_snap.get_output_for(obs_pl, action_pl, reuse=True)

        actions = {
            'x':[],
            'y':[],
            'z':[]
        }

        for ind, di in enumerate(data):
            print("evaluation {}".format(ind))
            start = di['start']
            next = di['next']
            jpos = di['jpos']
            quat = di['quat']
            obs = di['obs']
            help = ik_helper(jpos, quat)

            # setting up actions
            rad = np.linalg.norm(start - next)
            points = fibonacci_sphere(rad, NUMPTS)
            x_actions = points[:, 0] + start[0]
            y_actions = points[:, 1] + start[1]
            z_actions = points[:, 2] + start[2]
            actions['x'].append(x_actions)
            actions['y'].append(y_actions)
            actions['z'].append(z_actions)

            # load network from pkl file, snapshot['qf1']
            # snapshot = joblib.load('log/prim/reach/2019-08-20-15-52-39-191438-PDT/itr_2000.pkl')

            # discretize xyz somehow
            # pass into IK to get action
            q_vals = []
            for i in trange(len(x_actions)):
                xyz_action = np.array([x_actions[i],
                                       y_actions[i],
                                       z_actions[i]])

                # using IK_helper

                vel_action = help.xyz_to_jvel(xyz_action, jpos, quat)

                # pass into network, get single value?
                feed = {
                    obs_pl: obs.reshape(1, -1),
                    action_pl: vel_action.reshape(1, -1),
                }
                q_vals.append(session.run(qf1, feed))

            res.append(np.array(q_vals))

    return res, actions


def graph(fig, ax, data):
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    f = open('traj_dict.pkl', 'rb')
    traj = pickle.load(f)

    actions = traj['actions'] # use [0:7], gripper action included
    obss = traj['observations'] # [0:3] is eef pos, [8:11] is the goal
    rwrds = traj['rewards']
    j_poss = traj['j_positions']
    r_quats = traj['r_quats']

    data = []
    for i in np.arange(1,249, 5):
    # for i in np.arange((12*5) + 1, (20*5) + 1, 5):
        dic = {
            'start': obss[i][0:3],
            'next': obss[i+1][0:3],
            'jpos': j_poss[i],
            'quat': r_quats[i],
            'obs': obss[i]
        }
        data.append(dic)


    qvals, points = get_qvals(args.file, obss[0].size, actions[0].size, data)
    qvals = np.concatenate(qvals)
    norm_q = (qvals - np.mean(qvals))/np.std(qvals)

    fig = plt.figure()
    ax = Axes3D(fig)

    colormap = plt.cm.viridis  # or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    # plt.scatter(x, y, c=z, s=5, cmap=colormap, norm=normalize, marker='*')

    p = ax.scatter(points['x'][0], points['y'][0], points['z'][0], c=norm_q[0:NUMPTS], cmap=colormap, norm=normalize)
    # fig.colorbar(p)
    # plt.colorbar()
    def animate(i):
        ax.clear()
        xs = np.concatenate(points['x'][0:i+1])
        ys = np.concatenate(points['y'][0:i+1])
        zs = np.concatenate(points['z'][0:i+1])
        p = ax.scatter(xs, ys, zs, c=norm_q[0:NUMPTS*(i+1)], cmap=colormap, norm=normalize)


    anim = FuncAnimation(
        fig, animate, interval=500, frames=len(points['x']))

    plt.draw()
    plt.show()