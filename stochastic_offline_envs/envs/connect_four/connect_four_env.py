import numpy as np
import random
import gym
from gym import spaces

class ConnectFourEnv(gym.Env):

	def __init__(self, opponent_policy):
		self.board = ConnectFourBoard()
		self.observation_space = spaces.Box(low=0, 
			                                high=1, 
			                                shape=(2, self.board.width, self.board.height), dtype=np.int8)
		self.action_space = spaces.Discrete(self.board.width)
		self.opponent_policy = opponent_policy
		self.opp_policy_info = None

	def step(self, action):
		# Should always be the acting player's turns
		real_act = self.board.place(action, self.current_player)
		self.move_str = self.move_str + str(int(real_act + 1))
		done, winner = self.board.is_done()
		if done:
			reward = self._reward_from_winner(winner)
			obs = {'grid': self.board.get_grid(),
			       'move_str': self.move_str}
			return obs, reward, done, {}
		self.current_player = 1 - self.current_player
		self.opponent_step()
		done, winner = self.board.is_done()
		obs = {'grid': self.board.get_grid(),
			   'move_str': self.move_str}
		if done:
			reward = self._reward_from_winner(winner)
			return obs, reward, done, {'opp_policy_info': self.opp_policy_info}
		return obs, 0, done, {'opp_policy_info': self.opp_policy_info}

	def opponent_step(self):
		obs = {'grid': self.board.get_grid(),
			   'move_str': self.move_str}
		action, self.opp_policy_info = self.opponent_policy.sample(obs, 0, self.t)
		real_act = self.board.place(action, self.current_player)
		self.move_str = self.move_str + str(int(real_act + 1))
		self.current_player = 1 - self.current_player

	def reset(self):
		self.move_str = ''
		self.board = ConnectFourBoard()
		self.current_player = 0
		self.opponent_player = 1
		# self.opponent_player = np.random.randint(2)
		self.t = 0
		self.opponent_policy.reset()
		if self.current_player == self.opponent_player:
			self.opponent_step()
		obs = {'grid': self.board.get_grid(),
			   'move_str': self.move_str}
		return obs

	def render(self):
		print("=" * 20)
		print("O: Player 0, X: Player 1")
		print("Last move played by:", 1 - self.current_player)
		print("=" * 20)

		print(self.board.render_str())
		print("=" * 20)
		print(''.join([str(i) for i in range(self.board.width)]))
		print("=" * 20)

	def _reward_from_winner(self, winner):
		if winner == self.current_player:
			return 1
		elif winner == 1 - self.current_player:
			return -1
		return 0

class ConnectFourBoard:

	def __init__(self):
		self.width, self.height = 7, 6
		self.board = [[] for _ in range(self.width)]

	def place(self, col, color):
		assert col in range(self.width) and color in [0, 1]
		if self.full(col):
			# print('invalid action')
			col = 0
			while self.full(col):
				col += 1
		self.board[col].append(color)
		return col

	def render_str(self):
		r_str = ''
		for row in range(self.height-1, -1, -1):
			for col in range(self.width): 
				if len(self.board[col]) <= row:
					r_str += '-'
				else:
					r_str += 'O' if self.board[col][row] == 0 else 'X'
			r_str += '\n'
		return r_str

	def full(self, col):
		return len(self.board[col]) == self.height

	def get_grid(self):
		grid = -1 * np.ones((self.width, self.height))
		for i, col in enumerate(self.board):
			grid[i, :len(col)] = col
		one_hot_grid = np.zeros((2, self.width, self.height))
		one_hot_grid[0, grid==0] = 1
		one_hot_grid[1, grid==1] = 1
		return one_hot_grid

	def is_valid_and_consistent(self, col, color, target_env):
		if col not in range(self.width) or color not in [0, 1]:
			return False

		if self.full(col):
			return False

		y = len(self.board[col])

		if len(target_env.board.board[col]) <= y:
			return False

		return target_env.board.board[col][y] == color

	def is_done(self):
		# perhaps not as efficient as possible :)
		grid = self.get_grid()
		if np.sum(grid) == self.width * self.height:
			return True, 2
		for player in [0, 1]:
			for i in range(self.width):
				for j in range(self.height):
					pieces = []
					for c in range(4):
						if j + c < self.height:
							pieces.append(grid[player, i, j + c])
						else:
							pieces.append(0)
					if all(pieces):
						return True, player

					pieces = []
					for c in range(4):
						if i + c < self.width:
							pieces.append(grid[player, i + c, j])
						else:
							pieces.append(0)
					if all(pieces):
						return True, player

					pieces = []
					for c in range(4):
						if i + c < self.width and j + c < self.height:
							pieces.append(grid[player, i + c, j + c])
						else:
							pieces.append(0)
					if all(pieces):
						return True, player

					pieces = []
					for c in range(4):
						if i + c < self.width and j - c >= 0:
							pieces.append(grid[player, i + c, j - c])
						else:
							pieces.append(0)
					if all(pieces):
						return True, player
		return False, None

class GridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=0, 
                                            high=1, 
                                            shape=[2 * env.board.width * env.board.height], dtype=np.int8)
    
    def observation(self, obs):
        return obs['grid'].reshape(-1)