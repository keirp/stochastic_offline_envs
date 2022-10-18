from stochastic_offline_envs.policies.base import BasePolicy, PolicyStep
from collections import namedtuple
# from subprocess import Popen, PIPE, STDOUT
import pexpect

PolicyInfo = namedtuple("PolicyInfo", ["score"])


class C4Optimal(BasePolicy):

    def __init__(self, exec_dir='esper_envs/connect4'):
        self._name = 'c4optimal'
        self._exec_dir = exec_dir
        self.actions = range(7)
        self.c = pexpect.spawn('bash')
        self.c.sendline(f'cd {self._exec_dir};./c4solver -a')
        self.c.expect('done\r\n')

    def reset(self):
        pass

    def sample(self, obs, reward, t):
        move_str = obs['move_str']
        best_move = None
        best_score = None
        scores = self.scores_for_pos(move_str)
        for action, score in enumerate(scores):
            if best_score is None or score > best_score:
                best_score = score
                best_move = action
        return PolicyStep(action=best_move, info=PolicyInfo(score=best_score))

    def scores_for_pos(self, position):
        # send data to the solver
        self.c.sendline(str(position))
        self.c.expect('\r\n')
        self.c.expect('\r\n')
        solver_output = self.c.before.decode('utf-8')
        try:
            scores = [int(score) for score in solver_output.split(' ')[1:]]
        except:
            scores = None
        return scores
