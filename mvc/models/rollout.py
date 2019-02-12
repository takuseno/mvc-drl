import numpy as np

from mvc.preprocess import compute_returns, compute_gae
from mvc.misc.assertion import assert_batch_size_match, assert_shape_length


class Rollout:
    def __init__(self):
        self.flush()

    def add(self, obs_t, action_t, reward_t, value_t, log_prob_t, terminal_t):
        assert_batch_size_match(obs_t, action_t)
        assert_batch_size_match(obs_t, reward_t)
        assert_batch_size_match(obs_t, value_t)
        assert_batch_size_match(obs_t, log_prob_t)
        assert_batch_size_match(obs_t, terminal_t)
        assert_shape_length(reward_t, 1)
        assert_shape_length(value_t, 1)
        assert_shape_length(log_prob_t, 1)
        assert_shape_length(terminal_t, 1)

        self.obs_t.append(obs_t)
        self.actions_t.append(action_t)
        self.rewards_t.append(reward_t)
        self.values_t.append(value_t)
        self.log_probs_t.append(log_prob_t)
        self.terminals_t.append(terminal_t)

    def flush(self):
        self.obs_t = []
        self.actions_t = []
        self.rewards_t = []
        self.values_t = []
        self.log_probs_t = []
        self.terminals_t = []

    def fetch(self, gamma, lam):
        assert self.size() > 1

        step_length = self.size() - 1
        obs_t = np.array(self.obs_t)[:step_length]
        actions_t = np.array(self.actions_t)[:step_length]
        rewards_tp1 = np.array(self.rewards_t)[1:step_length + 1]
        terminals_tp1 = np.array(self.terminals_t)[1:step_length + 1]
        values_t = np.array(self.values_t)[:step_length]
        log_probs_t = np.array(self.log_probs_t)[:step_length]
        bootstrap_value = self.values_t[step_length]

        returns_t = compute_returns(bootstrap_value, rewards_tp1,
                                    terminals_tp1, gamma)
        advs_t = compute_gae(bootstrap_value, rewards_tp1, values_t,
                             terminals_tp1, gamma, lam)

        # normalize advantage
        advs_t = (advs_t - np.mean(advs_t)) / (np.std(advs_t) + 1e-8)

        return {
            'obs_t': obs_t,
            'actions_t': actions_t,
            'log_probs_t': log_probs_t,
            'returns_t': returns_t,
            'advantages_t': advs_t,
            'values_t': values_t
        }

    def size(self):
        # need reward and terminal at t+1
        return len(self.obs_t)
