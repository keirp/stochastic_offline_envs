from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
import gym_2048
import gym
from gym import spaces
import numpy as np
from collections import namedtuple
from gym.wrappers.time_limit import TimeLimit

wandb_run = 'keirp/stoch_rvs/2xz559rf'

PolicyInfo = namedtuple("PolicyInfo", [])


class TFEOfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('2048_5m_4x4.ds'), horizon=500,
                 n_interactions=int(1e5)):
        env_cls = lambda: TimeLimit(Success2048Wrapper(OneHot2048Wrapper(
            gym.make('2048-v0')), tile=128), max_episode_steps=horizon)

        def data_policy_fn():
            raise AttributeError(
                'Environment is attempting to regenerate data, which is not supported in this release. Double check the dataset path.')

        super().__init__(path, env_cls, data_policy_fn, horizon, n_interactions)


class OneHot2048Wrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_opts = 8

        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(
                                                self.width, self.height, self.n_opts),
                                            dtype=np.uint8)

    def observation(self, obs):
        flat_obs = obs.reshape(-1).copy()
        log_flat_obs = np.log2(flat_obs).astype(int)
        flat_obs[flat_obs != 0] = log_flat_obs[flat_obs != 0]
        one_hot = np.zeros((flat_obs.size, self.n_opts))
        one_hot[np.arange(flat_obs.size), flat_obs] = 1
        return one_hot.reshape((obs.shape[0], obs.shape[1], self.n_opts)).astype(np.uint8)


class Success2048Wrapper(gym.Wrapper):
    def __init__(self, env, tile=32):
        super().__init__(env)
        self.env = env
        self.tile = tile

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if r >= self.tile:
            r = 1
            d = True
        else:
            r = 0
        return s, r, d, i
