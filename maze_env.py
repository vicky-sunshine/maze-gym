import gym
import numpy as np
import time

MAZE_MAP_PATH = "maze_map/maze10.csv"


class Maze(gym.Env):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'right', 'left']
        self.n_actions = len(self.action_space)
        self._build_maze()
        self.n_features = self.MAZE_H * self.MAZE_W
        print(self.n_features)

    def _build_maze(self):
        self.maze = np.genfromtxt(MAZE_MAP_PATH, delimiter=',', dtype='int32')
        self.MAZE_H = len(self.maze)
        self.MAZE_W = len(self.maze[0])

        # find locattion of adventure
        x, y = np.where(self.maze == 1)
        self.user_loc = (1, 0)

    def _reset(self):
        self._build_maze()
        self.step_num = 0
        next_maze_state = np.copy(self.maze)
        next_maze_state[self.user_loc[0]][self.user_loc[1]] = 1
        return next_maze_state.reshape(1, self.MAZE_H*self.MAZE_W)

    def _step(self, action):
        self.step_num += 1
        user_loc_next = self.user_loc

        print("choose action: " + str(self.action_space[action]))
        if action == 0:   # up
            user_loc_next = (max(user_loc_next[0] - 1, 0), user_loc_next[1])
        elif action == 1:   # down
            user_loc_next = (min(user_loc_next[0] + 1, self.MAZE_H - 1), user_loc_next[1])
        elif action == 2:   # right
            user_loc_next = (user_loc_next[0], min(user_loc_next[1] + 1, self.MAZE_W - 1))
        elif action == 3:   # left
            user_loc_next = (user_loc_next[0], max(user_loc_next[1] - 1, 0))

        # reward function
        if self.maze[user_loc_next[0]][user_loc_next[1]] == 3:
            # get treasure
            print("=====================> Goal")
            reward = 1
            done = True
        elif self.maze[user_loc_next[0]][user_loc_next[1]] == 2:
            # jmp to hole
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        self.user_loc = user_loc_next
        next_maze_state = np.copy(self.maze)
        next_maze_state[user_loc_next[0]][user_loc_next[1]] = 1
        next_maze_state = next_maze_state.reshape(1, self.MAZE_H*self.MAZE_W)

        return next_maze_state, reward, done

    def _render(self):
        next_maze_state = np.copy(self.maze)
        next_maze_state[self.user_loc[0]][self.user_loc[1]] = 1
        print('step: ' + str(self.step_num))
        print(next_maze_state)
        print("======")
        # time.sleep(0.2)

if __name__ == '__main__':
    env = Maze()
