import numpy as np

from mvc.models.metrics import Metrics
from mvc.models.rollout import Rollout
from mvc.models.networks.base_network import BaseNetwork
from mvc.controllers.base_controller import BaseController
from mvc.preprocess import compute_returns, compute_gae


class PPOController(BaseController):
    def __init__(self,
                 network,
                 rollout,
                 metrics,
                 num_envs,
                 time_horizon,
                 epoch,
                 batch_size,
                 gamma,
                 lam,
                 log_interval=None,
                 final_steps=10 ** 6):
        assert isinstance(network, BaseNetwork)
        assert isinstance(rollout, Rollout)
        assert isinstance(metrics, Metrics)

        self.network = network
        self.rollout = rollout
        self.metrics = metrics
        self.num_envs = num_envs
        self.time_horizon = time_horizon
        self.epoch = epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.log_interval = time_horizon if not log_interval else log_interval
        self.final_steps = final_steps

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
        self.metrics.add('step', self.num_envs)
        for i in range(self.num_envs):
            if done[i] == 1.0:
                self.metrics.add('reward', info[i]['reward'])

        return output.action

    def should_update(self):
        return self.rollout.size() - 1 == self.time_horizon

    def update(self):
        assert self.should_update()

        # create batch from stored trajectories
        batches = self._batches()

        # flush stored trajectories
        self.rollout.flush()

        # update parameter
        losses = []
        for _ in range(self.epoch):
            for batch in batches:
                loss = self.network.update(**batch)
                losses.append(loss)
        mean_loss = np.mean(losses)

        # record metrics
        self.metrics.add('loss', mean_loss)

        return mean_loss

    def should_log(self):
        return self.metrics.get('step') % self.log_interval == 0

    def log(self):
        step = self.metrics.get('step')
        self.metrics.log_metric('reward', step)
        self.metrics.log_metric('loss', step)

    def stop_episode(self, obs, reward, info):
        pass

    def is_finished(self):
        return self.metrics.get('step') >= self.final_steps

    def _batches(self):
        assert self.rollout.size() > 1

        trajectory = self.rollout.fetch()
        step_length = self.rollout.size() - 1
        obs_t = trajectory['obs_t'][:step_length]
        actions_t = trajectory['actions_t'][:step_length]
        log_probs_t = trajectory['log_probs_t'][:step_length]
        values_t = trajectory['values_t'][:step_length]
        rewards_tp1 = trajectory['rewards_t'][1:step_length + 1]
        terminals_tp1 = trajectory['terminals_t'][1:step_length + 1]
        bootstrap_value = trajectory['values_t'][step_length]

        returns_t = compute_returns(bootstrap_value, rewards_tp1,
                                    terminals_tp1, self.gamma)
        advantages_t = compute_gae(bootstrap_value, rewards_tp1, values_t,
                                   terminals_tp1, self.gamma, self.lam)

        # flatten
        data_size = self.time_horizon * self.num_envs
        flat_obs_t = np.reshape(obs_t, (data_size,) + obs_t.shape[2:])
        flat_actions_t = np.reshape(actions_t, (data_size, -1))
        flat_log_probs_t = np.reshape(log_probs_t, (data_size, -1))
        flat_returns_t = np.reshape(returns_t, (-1,))
        flat_advantages_t = np.reshape(advantages_t, (-1,))

        # shuffle
        indices = np.random.permutation(np.arange(data_size))
        shuffled_obs_t = flat_obs_t[indices]
        shuffled_actions_t = flat_actions_t[indices]
        shuffled_log_probs_t = flat_log_probs_t[indices]
        shuffled_returns_t = flat_returns_t[indices]
        shuffled_advantages_t = flat_advantages_t[indices]

        # create batch data
        batches = []
        for i in range(data_size // self.batch_size):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            batch = {
                'obs_t': shuffled_obs_t[start:end],
                'actions_t': shuffled_actions_t[start:end],
                'log_probs_t': shuffled_log_probs_t[start:end],
                'returns_t': shuffled_returns_t[start:end],
                'advantages_t': shuffled_advantages_t[start:end]
            }
            batches.append(batch)

        return batches
