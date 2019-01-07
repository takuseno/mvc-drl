import numpy as np


class Rollout:
    def __init__(self):
        self.obs_t = []
        self.actions_t = []
        self.rewards_t = []
        self.values_t = []
        self.log_probs_t = []
        self.terminals_t = []

    def add(self, obs_t, action_t, reward_t, value_t, log_prob_t, terminal_t):
        assert obs_t.shape[0] == action_t.shape[0]
        assert obs_t.shape[0] == reward_t.shape[0]
        assert obs_t.shape[0] == value_t.shape[0]
        assert obs_t.shape[0] == log_prob_t.shape[0]
        assert obs_t.shape[0] == terminal_t.shape[0]
        assert len(reward_t.shape) == 1
        assert len(value_t.shape) == 1
        assert len(log_prob_t.shape) == 1
        assert len(terminal_t.shape) == 1

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

    def fetch(self):
        return {
            'obs_t': np.array(self.obs_t),
            'actions_t': np.array(self.actions_t),
            'rewards_t': np.array(self.rewards_t),
            'terminals_t': np.array(self.terminals_t),
            'values_t': np.array(self.values_t),
            'log_probs_t': np.array(self.log_probs_t)
        }

    def size(self):
        # need reward and terminal at t+1
        return len(self.obs_t)
