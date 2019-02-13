import numpy as np

from mvc.controllers.base_controller import BaseController
from mvc.models.metrics import Metrics
from mvc.models.networks.base_network import BaseNetwork


class EvalController(BaseController):
    def __init__(self, network, metrics, num_episodes):
        assert isinstance(network, BaseNetwork)
        assert isinstance(metrics, Metrics)
        assert metrics.has('step'), 'metrics must have step metrics'

        self.network = network
        self.metrics = metrics
        self.num_episodes = num_episodes

        self.metrics.register('eval_reward', 'queue')
        self.metrics.register('eval_episode', 'single')

        super().__init__(metrics, None, None, None, None)

    def _record_batch_rewards(self, done, info):
        for i in range(done.shape[0]):
            episode = self.metrics.get('eval_episode')
            if done[i] == 1.0 and episode < self.num_episodes:
                self.metrics.add('eval_episode', 1)
                self.metrics.add('eval_reward', info[i]['reward'])

    def step(self, obs, reward, done, info):
        output = self.network.infer(obs_t=obs)

        # record metrics for batch training
        if isinstance(done, np.ndarray):
            self._record_batch_rewards(done, info)

        return output.action

    # batch training will not call stop_episode
    def stop_episode(self, obs, reward, info):
        self.metrics.add('eval_reward', info['reward'])
        self.metrics.add('eval_episode', 1)

    def should_update(self):
        return False

    def update(self):
        raise Exception('EvalController does not update parameters')

    def should_log(self):
        return False

    def log(self):
        raise Exception('EvalController does not log intermediate data')

    def is_finished(self):
        is_finished = self.metrics.get('eval_episode') >= self.num_episodes
        if is_finished:
            step = self.metrics.get('step')
            self.metrics.log_metric('eval_reward', step)
            self.metrics.reset('eval_episode')
            self.metrics.reset('eval_reward')
        return is_finished

    def should_save(self):
        return False

    def should_eval(self):
        return False

    def save(self):
        pass
