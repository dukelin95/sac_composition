import numpy as np
import os
import h5py

from rllab.core.serializable import Serializable

from .replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size, seq_len):
        super(SimpleReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._sub_level_actions = np.zeros((max_replay_buffer_size,seq_len, self._action_dim))
        self._sub_level_probs = np.zeros((max_replay_buffer_size,seq_len, 1))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation,sub_level_actions,sub_level_probs, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._sub_level_actions[self._top] = sub_level_actions
        self._sub_level_probs[self._top] = sub_level_probs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            sub_level_actions=self._sub_level_actions[indices],
            sub_level_probs=self._sub_level_probs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(SimpleReplayBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            sa=self._sub_level_actions.tobytes(),
            sp=self._sub_level_probs.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        return d

    def __setstate__(self, d):
        super(SimpleReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._sub_level_actions = np.fromstring(d['sa']).reshape(self._max_buffer_size,seq_len, -1)
        self._sub_level_probs = np.fromstring(d['sp']).reshape(self._max_buffer_size,seq_len, -1)
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']


class DemoReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size, seq_len):
        super(DemoReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._sub_level_actions = np.zeros((max_replay_buffer_size,seq_len, self._action_dim))
        self._sub_level_probs = np.zeros((max_replay_buffer_size,seq_len, 1))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

        self.demo_folder = './demonstrations/'
        self.demo_buffer = SimpleReplayBuffer(self._env_spec, max_replay_buffer_size=5e4,seq_len=seq_len)
        self.demo_ratio = 1 # number of samples per batch

    def add_demos(self, sub_level_policies, g):
        # load demonstrations
        for folder in [directory for directory in os.listdir(self.demo_folder)]:

            hdf5_path = os.path.join(self.demo_folder, folder, "demo.hdf5")
            f = h5py.File(hdf5_path, "r")
            demos = list(f["data"].keys())
            for ep in demos:
                actions = f["data/{}/action".format(ep)].value
                rewards = f["data/{}/reward".format(ep)].value
                observations = f["data/{}/observation".format(ep)].value
                terminals = f["data/{}/terminal".format(ep)].value

                for t in range(100, len(actions)):
                    current_observation = observations[t-1]
                    if g != 0:
                        sub_level_obs = current_observation[:-g]
                    else:
                        sub_level_obs = current_observation

                    sub_level_actions = []
                    sub_level_probs = []

                    for j in range(len(sub_level_policies)):
                        sub_action, _ = sub_level_policies[j].get_action(sub_level_obs)
                        sub_level_actions.append(sub_action.reshape(1, -1))
                        pi = np.exp(sub_level_policies[j].log_pis_for(sub_level_obs.reshape(1, -1)))
                        # pi = np.exp(self.sub_level_policies[i].log_pis_for(sub_level_obs))
                        sub_level_probs.append(pi.reshape(1, -1))

                    sub_level_actions = np.stack(sub_level_actions, axis=0)
                    sub_level_actions = np.transpose(sub_level_actions, (1, 0, 2))
                    sub_level_probs = np.stack(sub_level_probs, axis=0)
                    sub_level_probs = np.transpose(sub_level_probs, (1, 0, 2))

                    action = actions[t]
                    reward = rewards[t]
                    terminal = terminals[t]
                    next_observation = observations[t]

                    self.demo_buffer.add_sample(
                        observation=current_observation,
                        sub_level_actions=sub_level_actions[0],
                        sub_level_probs=sub_level_probs[0],
                        action=action,
                        reward=reward,
                        terminal=terminal,
                        next_observation=next_observation)

                    if reward > 0.0: break
            f.close()

    def add_sample(self, observation,sub_level_actions,sub_level_probs, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._sub_level_actions[self._top] = sub_level_actions
        self._sub_level_probs[self._top] = sub_level_probs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size - self.demo_ratio)
        demo_batch = self.demo_buffer.random_batch(self.demo_ratio)
        return dict(
            observations=np.concatenate((self._observations[indices], demo_batch['observations'])),
            sub_level_actions=np.concatenate((self._sub_level_actions[indices], demo_batch['sub_level_actions'])),
            sub_level_probs=np.concatenate((self._sub_level_probs[indices], demo_batch['sub_level_probs'])),
            actions=np.concatenate((self._actions[indices], demo_batch['actions'])),
            rewards=np.concatenate((self._rewards[indices], demo_batch['rewards'])),
            terminals=np.concatenate((self._terminals[indices], demo_batch['terminals'])),
            next_observations=np.concatenate((self._next_obs[indices], demo_batch['next_observations'])),
        )

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(DemoReplayBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            sa=self._sub_level_actions.tobytes(),
            sp=self._sub_level_probs.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        return d

    def __setstate__(self, d):
        super(DemoReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._sub_level_actions = np.fromstring(d['sa']).reshape(self._max_buffer_size,seq_len, -1)
        self._sub_level_probs = np.fromstring(d['sp']).reshape(self._max_buffer_size,seq_len, -1)
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']