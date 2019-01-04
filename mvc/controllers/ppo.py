from mvc.controllers.base_controller import BaseController


class PPOController(Controller):
    def __init__(self, network, rollout, time_horizon, gamma, lam)
        self.network = network
        self.rollout = rollout
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.lam = lam

    def step(self, obs, reward, done):
        # infer action, policy, value
        action, log_prob, value = self.network.infer({'obs_t': obs})
        # store trajectory
        self.rollout.add(obs, action, reward, value, log_prob, done)
        return action

    def should_update(self):
        return self.rollout.size() == self.time_horizon

    def update(self)
        # create batch from stored trajectories
        batch = self.rollout.fetch(self.time_horizon, self.gamma, self.lam)
        # flush stored trajectories
        self.rollout.flush(self.time_horizon)
        # update parameter
        return self.network.update(**batch)

    def stop_episode(self, obs, reward):
        pass
