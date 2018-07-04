import gym
import numpy as np
import time

MAZE_MAP_PATH = "maze_map/maze44.csv"


class Maze(gym.Env):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'right', 'left']
        self.n_actions = len(self.action_space)
        self._build_maze()

    def _build_maze(self):
        self.maze = np.genfromtxt(MAZE_MAP_PATH, delimiter=',', dtype='int32')
        self.MAZE_H = len(self.maze)
        self.MAZE_W = len(self.maze[0])

        # find locattion of adventure
        x, y = np.where(self.maze == 1)
        self.user_loc = (1, 0)

    def _reset(self):
        self._build_maze()
        return self.user_loc

    def _step(self, action):
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
            s_ = 'terminal'
        elif self.maze[user_loc_next[0]][user_loc_next[1]] == 2:
            # jmp to hole
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            s_ = user_loc_next
            reward = 0
            done = False
            self.user_loc = user_loc_next
        return s_, reward, done

    def _render(self):
        maze_next = np.copy(self.maze)
        maze_next[self.user_loc[0]][self.user_loc[1]] = 1
        print(maze_next)
        print("user loc: " + str(self.user_loc))
        print("======")
        time.sleep(0.2)

if __name__ == '__main__':
    env = Maze()
