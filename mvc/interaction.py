import numpy as np


def initial_inputs(env):
    obs = env.reset()
    reward = np.zeros((obs.shape[0],), dtype=np.float32)
    done = np.zeros((obs.shape[0],), dtype=np.float32)
    info = {}
    return obs, reward, done, info


def step(env, view, obs, reward, done, info):
    action = view.step(obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info


def loop(env, view, hook=None):
    obs, reward, done, info = initial_inputs(env)
    while True:
        obs, reward, done, info = step(env, view, obs, reward, done, info)

        if hook is not None:
            hook(view)

        if view.is_finished():
            break


class BatchInteraction:
    def __init__(self, env, view, eval_env=None, eval_view=None):
        self.env = env
        self.view = view
        self.eval_env = eval_env
        self.eval_view = eval_view

    def start(self, hook=None):
        def _hook(view):
            if view.should_eval():
                loop(self.eval_env, self.eval_view)

            if hook is not None:
                hook(view)

        loop(self.env, self.view, _hook)
