import numpy as np


class BatchEnvWrapper:
    def __init__(self, envs, render=False):
        self.envs = envs
        self.render = render
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.sum_of_rewards = [0.0 for _ in envs]

    def step(self, action):
        obs_t = []
        rewards_t = []
        dones_t = []
        infos_t = []
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(action[i])
            self.sum_of_rewards[i] += reward
            if done:
                obs = env.reset()
                info['reward'] = self.sum_of_rewards[i]
                self.sum_of_rewards[i] = 0.0
            done = 1.0 if done else 0.0
            obs_t.append(obs)
            rewards_t.append(reward)
            dones_t.append(done)
            infos_t.append(info)
        if self.render:
            self.envs[0].render()
        return np.array(obs_t), np.array(rewards_t), np.array(dones_t), infos_t

    def reset(self):
        obs_t = []
        for env in self.envs:
            obs_t.append(env.reset())
        return np.array(obs_t)
