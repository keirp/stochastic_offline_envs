from stochastic_offline_envs.policies.base import BasePolicy, PolicyStep
import numpy as np
from collections import namedtuple

PolicyInfo = namedtuple("PolicyInfo", ['policy_idx'])


class EpisodicMixturePolicy(BasePolicy):
    def __init__(self, policies, ps):
        self._name = 'mixture'
        self.policies = policies
        self.ps = ps

    def reset(self):
        self.policy_idx = np.random.choice(len(self.policies), p=self.ps)
        self.policy = self.policies[self.policy_idx]
        self.policy.reset()

    def sample(self, obs, reward, t):
        action, info = self.policy.sample(obs, reward, t)
        return PolicyStep(action=action, info=PolicyInfo(policy_idx=self.policy_idx))


class StateMixturePolicy(BasePolicy):
    def __init__(self, policies, ps):
        self._name = 'mixture'
        self.policies = policies
        self.ps = ps

    def reset(self):
        for policy in self.policies:
            policy.reset()

    def sample(self, obs, reward, t):
        policy_steps = [policy.sample(obs, reward, t)
                        for policy in self.policies]
        idx = np.random.choice(range(len(policy_steps)), p=self.ps)
        return policy_steps[idx]
