import gym
import numpy as np
from gym import spaces


class GamblingEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.uint8)
        self.state = 0

    def get_obs(self):
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        return self.get_obs()

    def step(self, action):
        if self.state == 0:
            reward = 0
            done = False
            if action == 0:
                if np.random.random_sample() < 0.5:  # -2.5
                    self.state = 1
                else:
                    self.state = 2
            elif action == 1:
                if np.random.random_sample() < 0.5:  # (-10 + 1)/2
                    self.state = 3
                else:
                    self.state = 4
            else:
                if np.random.random_sample() < 0.5:  # 1
                    self.state = 5
                else:
                    self.state = 6
        elif self.state == 1:
            reward = 5
            done = True
        elif self.state == 2:
            reward = -15
            done = True
        elif self.state == 3:
            reward = 1
            done = True
        elif self.state == 4:
            reward = -6
            done = True
        elif self.state == 5:
            reward = 1
            done = True
        elif self.state == 6:
            reward = 1
            done = True
        return self.get_obs(), reward, done, {}
