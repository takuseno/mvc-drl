import numpy as np

from mvc.models.metrics import Metrics
from mvc.models.rollout import Rollout
from mvc.models.networks.base_network import BaseNetwork
from mvc.controllers.base_controller import BaseController
from mvc.preprocess import compute_returns, compute_gae


def shuffle_batch(batch, size):
    indices = np.random.permutation(np.arange(size))
    # check first not to change original data in the error case
    for key in batch.keys():
        assert batch[key].shape[0] == size
    for key in batch.keys():
        batch[key] = batch[key][indices]
    return batch

class PPOController(BaseController):
    def __init__(self,
                 network,
                 rollout,
                 metrics,
                 time_horizon,
                 gamma,
                 lam,
                 log_interval=None):
        assert isinstance(network, BaseNetwork)
        assert isinstance(rollout, Rollout)
        assert isinstance(metrics, Metrics)

        self.network = network
        self.rollout = rollout
        self.metrics = metrics
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.lam = lam
        self.log_interval = time_horizon if not log_interval else log_interval

        self.metrics.register('step', 'single')
        self.metrics.register('loss', 'queue')
        self.metrics.register('reward', 'queue')

    def step(self, obs, reward, done, info):
        # infer action, policy, value
        output = self.network.infer(obs_t=obs)
        # store trajectory
        self.rollout.add(obs, output.action, reward,
                         output.value, output.log_prob, done)

        # record metrics
        self.metrics.add('step', obs.shape[0])
        for i in range(obs.shape[0]):
            if done[i]:
                self.metrics.add('reward', info[i]['reward'])

        return output.action

    def should_update(self):
        return self.rollout.size() - 1 == self.time_horizon

    def update(self):
        assert self.should_update()

        # create batch from stored trajectories
        batch = shuffle_batch(self._batch(), self.time_horizon)
        # flush stored trajectories
        self.rollout.flush()
        # update parameter
        loss = self.network.update(**batch)

        # record metrics
        self.metrics.add('loss', loss)

        return loss

    def should_log(self):
        return self.metrics.get('step') % self.log_interval == 0

    def log(self):
        step = self.metrics.get('step')
        self.metrics.log_metric('reward', step)
        self.metrics.log_metric('loss', step)

    def stop_episode(self, obs, reward, info):
        pass

    def _batch(self):
        assert self.rollout.size() > 1

        trajectory = self.rollout.fetch()
        step_length = self.rollout.size() - 1
        values_t = trajectory['values_t'][:step_length]
        rewards_tp1 = trajectory['rewards_t'][1:step_length + 1]
        terminals_tp1 = trajectory['terminals_t'][1:step_length + 1]
        bootstrap_value = trajectory['values_t'][step_length]

        returns_t = compute_returns(bootstrap_value, rewards_tp1,
                                    terminals_tp1, self.gamma)
        advantages_t = compute_gae(bootstrap_value, rewards_tp1, values_t,
                                   terminals_tp1, self.gamma, self.lam)

        return {
            'obs_t': trajectory['obs_t'][:step_length],
            'actions_t': trajectory['actions_t'][:step_length],
            'log_probs_t': trajectory['log_probs_t'][:step_length],
            'returns_t': returns_t,
            'advantages_t': advantages_t
        }
