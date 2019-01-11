import numpy as np
import mvc.logger as logger


def log_metric(reward, step, episode):
    logger.log_metric('reward_in_step', reward, step)
    logger.log_metric('reward_in_episode', reward, episode)


class BatchInteraction:
    def __init__(self, env, view):
        self.env = env
        self.view = view

        # metrics
        self.sum_of_rewards = np.zeros((env.size(),), dtype=np.float32)
        self.t = 0
        self.episode = 0

    def loop(self):
        obs = self.env.reset()
        reward = np.zeros((obs.shape[0],), dtype=np.float32)
        done = np.zeros((obs.shape[0],), dtype=np.float32)
        while True:
            obs, reward, done = self.step(obs, reward, done)

    def step(self, obs, reward, done):
        action = self.view.step(obs, reward, done)
        obs, reward, done, _ = self.env.step(action)

        for i in range(obs.shape[0]):
            self.t += 1
            self.sum_of_rewards[i] += reward[i]
            if done[i] == 1.0:
                self.episode += 1
                log_metric(self.sum_of_rewards[i], self.t, self.episode)
                self.sum_of_rewards[i] = 0.0

        return obs, reward, done
