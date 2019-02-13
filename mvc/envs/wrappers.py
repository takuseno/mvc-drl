import numpy as np


class BatchEnvWrapper:
    def __init__(self, envs, render=False):
        self.envs = envs
        self.render = render
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.sum_of_rewards = [0.0 for _ in envs]

    def _step_single_env(self, index, action):
        env = self.envs[index]
        obs, reward, done, info = env.step(action[index])
        self.sum_of_rewards[index] += reward
        if done:
            obs = env.reset()
            info['reward'] = self.sum_of_rewards[index]
            self.sum_of_rewards[index] = 0.0
        done = 1.0 if done else 0.0
        return obs, reward, done, info

    def step(self, action):
        obs_t, rewards_t, dones_t, infos_t = [], [], [], []
        for i in range(len(self.envs)):
            obs, reward, done, info = self._step_single_env(i, action[i])
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


class MuJoCoWrapper:
    def __init__(self, env, reward_scale=1.0, render=False):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_scale = reward_scale
        self.render = render
        self.sum_of_rewards = 0.0

    def step(self, action):
        high = self.action_space.high
        obs, reward, done, info = self.env.step(action * high)

        if self.render:
            self.env.render()

        self.sum_of_rewards += reward
        if done:
            info['reward'] = self.sum_of_rewards

        return obs, reward * self.reward_scale, done, info

    def reset(self):
        self.sum_of_rewards = 0.0
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)
