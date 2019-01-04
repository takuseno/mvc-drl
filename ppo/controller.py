from ppo.models.rollout import Rollout
from ppo.models.function import Function


class Controller:
    def step(self, obs, reward, done):
        raise NotImplementedError('implement step function')

    def should_update(self):
        raise NotImplementedError('implement should_update function')

    def update(self):
        raise NotImplementedError('implement update function')


class PPOController(Controller):
    def __init__(self, function, time_horizon, )
        self.function = function
        self.time_horizon = time_horizon

        self.rollout = Rollout()

    def step(self, obs, reward, done):
        return self.function.infer(obs)

    def should_update(self):
        return self.rollout.size() == self.time_horizon

    def update(self)
        batch = self.rollout.fetch()
        self.rollout.clean()
        return self.function.update()
