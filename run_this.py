from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(1000):
        print("episode: " + str(episode))
        # initial observation
        observation = env._reset()

        while True:
            # fresh env
            env._render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env._step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update()
