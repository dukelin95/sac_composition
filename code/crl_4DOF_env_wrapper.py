from rllab.envs.env_spec import EnvSpec
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from rllab.misc import logger

from robosuite.wrappers import Wrapper
from cached_property import cached_property


import numpy as np

# make robosuite env to a gym env with compatibility with sac
class CRL4DOFWrapper(Wrapper, Serializable):

    def __init__(self, env, use_gripper):
        super().__init__(env)
        Serializable.quick_init(self, locals())
        self.keys = ["robot-state", "object-state"]

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low=low, high=high)
        low, high = self.env.action_spec

        self.use_gripper = use_gripper

        if self.use_gripper:
            self.action_space = Box(low=low[3:8], high=high[3:8])
        else:
            self.action_space = Box(low=low[3:7], high=high[3:7])

        self.spec = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )


    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])

        # for composition rl
        # ob_lst.append(self.env.goal)
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        if self.use_gripper:
            action = np.concatenate([np.zeros(3), action])
        else:
            action = np.concatenate([np.zeros(3), action, -np.ones(1)])

        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def terminate(self):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def log_diagnostics(self, paths):
        if hasattr(self.env, 'instructive'):
            chance = self.env.instructive * (1 / (1 + self.env.decay * self.env.instructive_counter))
            logger.record_tabular('instructive_chance', chance)
            logger.record_tabular('instructive_count', self.env.instructive_counter)