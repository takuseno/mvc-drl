import numpy as np


class BatchInteraction:
    def __init__(self, env, view, eval_env=None, eval_view=None):
        self.env = env
        self.view = view
        self.eval_env = eval_env
        self.eval_view = eval_view

    def loop(self):
        obs, reward, done, info = self.initial_inputs(self.env)
        while True:
            obs, reward, done, info = self.step(self.env, self.view, obs,
                                                reward, done, info)
            if self.eval_view is not None and self.view.should_eval():
                self.eval_loop()

            if self.view.is_finished():
                break

    def eval_loop(self):
        obs, reward, done, info = self.initial_inputs(self.eval_env)
        while True:
            obs, reward, done, info = self.step(self.eval_env, self.eval_view,
                                                obs, reward, done, info)
            if self.eval_view.is_finished():
                break

    def initial_inputs(self, env):
        obs = env.reset()
        reward = np.zeros((obs.shape[0],), dtype=np.float32)
        done = np.zeros((obs.shape[0],), dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def step(self, env, view, obs, reward, done, info):
        action = view.step(obs, reward, done, info)
        obs, reward, done, info = env.step(action)
        return obs, reward, done, info
