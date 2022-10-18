from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.gambling.gambling import GamblingEnv
from stochastic_offline_envs.policies.random import RandomPolicy


class GamblingOfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('gambling.ds'), horizon=5,
                 n_interactions=int(1e5)):
        env_cls = lambda: GamblingEnv()

        def data_policy_fn():
            test_env = env_cls()
            test_env.action_space
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        super().__init__(path, env_cls, data_policy_fn, horizon, n_interactions)
