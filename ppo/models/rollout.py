import numpy as np

from ppo.preprocess import compute_returns, compute_gae


class Rollout:
    def __init__(self):
        self.obs_t = []
        self.actions_t = []
        self.rewards_t = []
        self.values_t = []
        self.log_probs_t = []
        self.terminals_t = []

    def add(self, obs_t, action_t, reward_t, value_t, log_prob_t, terminal_t):
        assert obs_t.shape[0] == action_t.shape[0],
            'all arguments must have same batch size'
        assert obs_t.shape[0] == reward_t.shape[0],
            'all arguments must have same batch size'
        assert obs_t.shape[0] == value_t.shape[0],
            'all arguments must have same batch size'
        assert obs_t.shape[0] == log_prob_t.shape[0],
            'all arguments must have same batch size'
        assert obs_t.shape[0] == terminal_t.shape[0],
            'all arguments must have same batch size'
        self.obs_t.append(obs_t)
        self.actions_t.append(action_t)
        self.rewards_t.append(reward_t)
        self.values_t.append(value_t)
        self.log_probs_t.append(log_prob_t)
        self.terminals_t.append(terminal_t)

    def flush(self, n):
        assert n <= self.size(), 'n must be smaller than stored transitions'
        self.obs_t = self.obs_t[n+1:]
        self.actions_t = self.actions_t[n+1:]
        self.rewards_t = self.rewards_t[n+1:]
        self.values_t = self.values_t[n+1:]
        self.log_probs_t = self.log_probs_t[n+1:]
        self.terminals_t = self.terminals_t[n+1:]

    def fetch(self, n, gamma, lam):
        values_t = np.array(self.values_t[:n])
        rewards_tp1 = np.array(self.rewards_t[1:1 + n])
        terminals_tp1 = np.array(self.terminals_t[1:1 + n])
        bootstrap_value = self.values_t[n]

        returns_t = compute_returns(rewards_tp1, bootstrap_value,
                                    terminals_tp1, gamma)
        advantages_t = compute_gae(bootstrap_value, rewards_tp1, values_t,
                                   terminals_tp1, gamma, lam)

        return {
            'obs_t': np.array(self.obs_t[:n]),
            'actions_t': np.array(self.actions_t[:n]),
            'log_probs_t': np.array(self.log_probs_t[:n]),
            'returns_t': returns_t,
            'advantages_t': advantages_t
        }

    def size(self):
        # need reward and terminal at t+1
        return len(self.obs_t) - 1
