import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.policies.base import Policy

# hyperparam
SIGMA = 0.005
MU = 0.09

class IK_Policy(Policy):

    def __init__(self, axis='x'):
        # TODO is this env_spec needed?
        super().__init__(env_spec = None)
        self.reach = False
        self.build()
        if axis == 'x':
            self.axis = 0
        elif axis == 'y':
            self.axis = 1
        elif axis == 'z':
            self.axis = 2
        elif axis == 'r':
            self.reach = True
        else:
            raise ValueError("Only x, y, z, or r (for reach) accepted")


    def build(self):
        self.mu = tf.placeholder(tf.float64, shape=(3,))
        self.sig = tf.placeholder(tf.float64, shape=(3,))
        self.dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sig)
        self.x_t = self.dist.sample()
        self.log_pi_t = self.dist.log_prob(self.x_t)
        self.actions = tf.tanh(tf.stop_gradient(self.x_t))

    def get_action(self, obs):
        assert(len(obs) == 11)
        # order: gripper_qpos, eef_pos, marker/cube_pos
        marker_pos = obs[5:8]
        eef_pos = obs[2:5]
        diff = marker_pos - eef_pos
        mu = np.zeros(3)

        if self.reach:
            mu = np.sign(diff) * MU
            sig = np.ones(3) * SIGMA
        else:
            mu[self.axis] = np.sign(diff[self.axis]) * MU
            sig = np.zeros(3) + 1e-6
            sig[self.axis] = SIGMA

        feed_dict = {
            self.mu: mu,
            self.sig: sig
        }
        actions = tf.get_default_session().run(self.actions, feed_dict)
        return actions, None

    def log_pis_for(self, obs):
        assert (len(obs) == 11)
        marker_pos = obs[5:8]
        eef_pos = obs[2:5]
        diff = marker_pos - eef_pos
        mu = np.zeros(3)

        if self.reach:
            mu = np.sign(diff) * MU
            sig = np.ones(3) * SIGMA
        else:
            mu[self.axis] = np.sign(diff[self.axis]) * MU
            sig = np.zeros(3) + 1e-6
            sig[self.axis] = SIGMA

        feed_dict = {
            self.mu: mu,
            self.sig: sig
        }
        log_pis = tf.get_default_session().run(self.log_pi_t, feed_dict)
        return log_pis