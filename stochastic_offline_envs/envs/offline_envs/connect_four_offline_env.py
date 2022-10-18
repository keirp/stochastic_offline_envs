from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.connect_four.connect_four_env import ConnectFourEnv
from stochastic_offline_envs.policies.random import RandomPolicy
from stochastic_offline_envs.policies.mixture_policy import EpisodicMixturePolicy, StateMixturePolicy
from stochastic_offline_envs.policies.c4_optimal import C4Optimal
from stochastic_offline_envs.policies.c4_exploitable import C4Specialized, C4MarkovExploitable
from gym import spaces
from pathlib import Path


class ConnectFourOfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('c4data_mdp.ds'), horizon=50,
                 n_interactions=int(1e6),
                 exec_dir=default_path('../connect4')):
        opp_policy = C4MarkovExploitable(exec_dir=exec_dir)
        env_cls = lambda: ConnectFourEnv(opp_policy)

        def data_policy_fn():
            raise AttributeError(
                'Environment is attempting to regenerate data, which is not supported in this release. Double check the dataset path.')
            # data_specialized_policy = C4Specialized()
            # data_eps_greedy = self._eps_greedy_policy(
            #     eps=0.01, exec_dir=exec_dir)
            # data_policy = EpisodicMixturePolicy(policies=[data_specialized_policy, data_eps_greedy],
            #                                     ps=[0.5, 0.5])
            # return data_policy

        super().__init__(path, env_cls, data_policy_fn, horizon, n_interactions)

    def _eps_greedy_policy(self, eps, exec_dir):
        optimal_policy = C4Optimal(exec_dir=exec_dir)
        action_space = spaces.Discrete(7)
        random_policy = RandomPolicy(action_space)

        eps_greedy_policy = StateMixturePolicy(policies=[optimal_policy, random_policy],
                                               ps=[1 - eps, eps])

        return eps_greedy_policy
