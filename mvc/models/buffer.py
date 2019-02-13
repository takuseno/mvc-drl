from collections import deque

import numpy as np


class Buffer:
    def __init__(self, maxlen=10 ** 6):
        self.obs_t = deque(maxlen=maxlen)
        self.actions_t = deque(maxlen=maxlen)
        self.rewards_t = deque(maxlen=maxlen)
        self.dones_t = deque(maxlen=maxlen)

    def add(self, obs_t, action_t, reward_t, done_t):
        self.obs_t.append(obs_t)
        self.actions_t.append(action_t)
        self.rewards_t.append(reward_t)
        self.dones_t.append(done_t)

    def reset(self):
        self.obs_t.clear()
        self.actions_t.clear()
        self.rewards_t.clear()
        self.dones_t.clear()

    def size(self):
        return len(self.obs_t)

    def fetch(self, batch_size):
        assert batch_size < self.size()

        indices = np.random.randint(self.size() - 1, size=batch_size)
        obs_t = []
        actions_t = []
        rewards_tp1 = []
        obs_tp1 = []
        dones_tp1 = []
        for index in indices:
            obs_t.append(self.obs_t[index])
            actions_t.append(self.actions_t[index])
            rewards_tp1.append(self.rewards_t[index + 1])
            obs_tp1.append(self.obs_t[index + 1])
            dones_tp1.append(self.dones_t[index + 1])

        return {
            'obs_t': np.array(obs_t),
            'actions_t': np.array(actions_t),
            'rewards_tp1': np.array(rewards_tp1),
            'obs_tp1': np.array(obs_tp1),
            'dones_tp1': np.array(dones_tp1)
        }
