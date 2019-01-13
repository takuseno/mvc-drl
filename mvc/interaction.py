import numpy as np


class BatchInteraction:
    def __init__(self, env, view):
        self.env = env
        self.view = view

    def loop(self):
        obs = self.env.reset()
        reward = np.zeros((obs.shape[0],), dtype=np.float32)
        done = np.zeros((obs.shape[0],), dtype=np.float32)
        info = {}
        while True:
            obs, reward, done, info = self.step(obs, reward, done, info)
            if self.view.is_finished():
                break

    def step(self, obs, reward, done, info):
        action = self.view.step(obs, reward, done, info)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
