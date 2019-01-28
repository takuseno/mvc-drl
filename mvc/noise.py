import numpy as np


# from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma, theta=.15, time=1e-2, init_x=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.time = time
        self.init_x = init_x
        self.prev_x = None
        self.reset()

    def __call__(self):
        normal = np.random.normal(size=self.mean.shape)
        new_x = self.prev_x + self.theta * (self.mean - self.prev_x) \
            * self.time + self.sigma * np.sqrt(self.time) * normal
        self.prev_x = new_x
        return new_x

    def reset(self):
        if self.init_x is not None:
            self.prev_x = self.init_x
        else:
            self.prev_x = np.zeros_like(self.mean)


# for evaluation and stochastic policy
class EmptyNoise:
    def __call__(self):
        return 0.0

    def reset(self):
        pass
