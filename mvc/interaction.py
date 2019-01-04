import numpy as np


class SerialInteraction:
    def __init__(self, env, view):
        self.env = env
        self.view = view

    def loop(self):
        while True:
            obs = self.env.reset()
            reward = 0.0
            done = False
            while not done:
                action = self.view.step(obs, reward)
                obs, reward, _, done = self.env.step(action)
            self.view.stop_episode(obs, reward)

class BatchInteraction:
    def __init__(self, env, view):
        self.env = env
        self.view = view

    def loop(self):
        obs = self.env.reset()
        reward = np.zeros((obs.shape[0],), dtype=np.float32)
        while True:
            action = self.view.step(obs, reward)
            obs, reward, _, _ = self.env.step(action)
